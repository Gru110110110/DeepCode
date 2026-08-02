// The restricted-token design follows OpenAI Codex's Apache-2.0 licensed
// Windows sandbox architecture, adapted to DeepCode's smaller execution API.

use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{c_void, OsStr};
use std::os::windows::ffi::OsStrExt;
use std::os::windows::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::ptr;
use std::time::Duration;

use deepcode_core::error::{DeepCodeError, Result};
use sha2::{Digest, Sha256};
use windows_sys::Win32::Foundation::{
    CloseHandle, GetLastError, LocalFree, SetHandleInformation, ERROR_SUCCESS, HANDLE,
    HANDLE_FLAG_INHERIT, HLOCAL, WAIT_OBJECT_0, WAIT_TIMEOUT,
};
use windows_sys::Win32::Security::Authorization::{
    ConvertStringSidToSidW, GetNamedSecurityInfoW, SetEntriesInAclW, SetNamedSecurityInfoW,
    EXPLICIT_ACCESS_W, SET_ACCESS, TRUSTEE_IS_SID, TRUSTEE_IS_UNKNOWN, TRUSTEE_W,
};
use windows_sys::Win32::Security::{
    AdjustTokenPrivileges, CopySid, CreateRestrictedToken, CreateWellKnownSid, GetLengthSid,
    GetTokenInformation, LookupPrivilegeValueW, SetTokenInformation, TokenDefaultDacl, TokenGroups,
    ACL, DACL_SECURITY_INFORMATION, PSID, SID_AND_ATTRIBUTES, SUB_CONTAINERS_AND_OBJECTS_INHERIT,
    TOKEN_ADJUST_DEFAULT, TOKEN_ADJUST_PRIVILEGES, TOKEN_ADJUST_SESSIONID, TOKEN_ASSIGN_PRIMARY,
    TOKEN_DUPLICATE, TOKEN_GROUPS, TOKEN_PRIVILEGES, TOKEN_QUERY,
};
use windows_sys::Win32::Storage::FileSystem::{
    ReadFile, DELETE, FILE_APPEND_DATA, FILE_DELETE_CHILD, FILE_GENERIC_EXECUTE, FILE_GENERIC_READ,
    FILE_GENERIC_WRITE,
};
use windows_sys::Win32::System::JobObjects::{
    CreateJobObjectW, JobObjectExtendedLimitInformation, SetInformationJobObject,
    TerminateJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_BREAKAWAY_OK,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
use windows_sys::Win32::System::Pipes::CreatePipe;
use windows_sys::Win32::System::Threading::{
    CreateProcessAsUserW, DeleteProcThreadAttributeList, GetCurrentProcess, GetExitCodeProcess,
    InitializeProcThreadAttributeList, UpdateProcThreadAttribute, WaitForSingleObject,
    CREATE_NO_WINDOW, CREATE_UNICODE_ENVIRONMENT, EXTENDED_STARTUPINFO_PRESENT, INFINITE,
    LPPROC_THREAD_ATTRIBUTE_LIST, PROCESS_INFORMATION, STARTF_USESTDHANDLES, STARTUPINFOEXW,
};

use crate::{FilesystemAccess, PreparedCommand};

const DISABLE_MAX_PRIVILEGE: u32 = 0x01;
const LUA_TOKEN: u32 = 0x04;
const WRITE_RESTRICTED: u32 = 0x08;
const WIN_WORLD_SID: i32 = 1;
const SE_GROUP_LOGON_ID: u32 = 0xC000_0000;
const PROC_THREAD_ATTRIBUTE_HANDLE_LIST: usize = 0x0002_0002;
const PROC_THREAD_ATTRIBUTE_JOB_LIST: usize = 0x0002_000D;
const WRITE_ALLOW_MASK: u32 =
    FILE_GENERIC_READ | FILE_GENERIC_WRITE | FILE_GENERIC_EXECUTE | DELETE;

pub(super) fn apply_no_network_environment(env: &mut BTreeMap<String, String>) {
    for (key, value) in [
        ("HTTP_PROXY", "http://127.0.0.1:9"),
        ("HTTPS_PROXY", "http://127.0.0.1:9"),
        ("ALL_PROXY", "http://127.0.0.1:9"),
        ("NO_PROXY", "localhost,127.0.0.1,::1"),
        ("GIT_HTTP_PROXY", "http://127.0.0.1:9"),
        ("GIT_HTTPS_PROXY", "http://127.0.0.1:9"),
        ("GIT_SSH_COMMAND", "cmd /c exit 1"),
        ("PIP_NO_INDEX", "1"),
        ("NPM_CONFIG_OFFLINE", "true"),
        ("CARGO_NET_OFFLINE", "true"),
        ("DEEPCODE_NETWORK_SANDBOX", "advisory"),
    ] {
        env.entry(key.into()).or_insert_with(|| value.into());
    }
}

pub(super) fn execute(
    prepared: &PreparedCommand,
    timeout: Duration,
) -> Result<std::process::Output> {
    unsafe { execute_inner(prepared, timeout) }
}

unsafe fn execute_inner(
    prepared: &PreparedCommand,
    timeout: Duration,
) -> Result<std::process::Output> {
    let write_roots = prepared
        .filesystem_rules
        .iter()
        .filter(|rule| matches!(rule.access, FilesystemAccess::Write) && rule.path.exists())
        .map(|rule| canonical_or_original(&rule.path))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let policy_key = prepared
        .filesystem_rules
        .iter()
        .map(|rule| format!("{:?}:{}", rule.access, path_key(&rule.path)))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>()
        .join("|");
    let mut capability_sids = if write_roots.is_empty() {
        vec![LocalSid::new(&capability_sid("deepcode-read-only-v1"))?]
    } else {
        write_roots
            .iter()
            .map(|root| LocalSid::new(&capability_sid(&format!("{}|{policy_key}", path_key(root)))))
            .collect::<Result<Vec<_>>>()?
    };

    for (root, sid) in write_roots.iter().zip(capability_sids.iter()) {
        set_path_ace(root, sid.as_ptr(), SET_ACCESS, WRITE_ALLOW_MASK)?;
    }
    for rule in prepared
        .filesystem_rules
        .iter()
        .filter(|rule| matches!(rule.access, FilesystemAccess::Deny) && rule.path.exists())
    {
        let denied = canonical_or_original(&rule.path);
        for (root, sid) in write_roots.iter().zip(capability_sids.iter()) {
            if denied.starts_with(root) || root.starts_with(&denied) {
                set_path_ace(
                    &denied,
                    sid.as_ptr(),
                    windows_sys::Win32::Security::Authorization::DENY_ACCESS,
                    FILE_GENERIC_WRITE | FILE_APPEND_DATA | FILE_DELETE_CHILD | DELETE,
                )?;
            }
        }
    }

    let token = create_restricted_token(&mut capability_sids)?;
    spawn_and_capture(prepared, token.0, timeout)
}

fn canonical_or_original(path: &Path) -> PathBuf {
    path.canonicalize().unwrap_or_else(|_| path.to_path_buf())
}

fn path_key(path: &Path) -> String {
    canonical_or_original(path)
        .to_string_lossy()
        .replace('\\', "/")
        .to_ascii_lowercase()
}

fn capability_sid(key: &str) -> String {
    let digest = Sha256::digest(key.as_bytes());
    let part = |offset: usize| {
        u32::from_le_bytes(digest[offset..offset + 4].try_into().expect("four bytes"))
    };
    format!("S-1-5-21-{}-{}-{}-{}", part(0), part(4), part(8), part(12))
}

struct LocalSid(PSID);

impl LocalSid {
    fn new(value: &str) -> Result<Self> {
        let mut sid = ptr::null_mut();
        let wide = to_wide(value);
        if unsafe { ConvertStringSidToSidW(wide.as_ptr(), &mut sid) } == 0 {
            return Err(last_error("ConvertStringSidToSidW"));
        }
        Ok(Self(sid))
    }

    fn as_ptr(&self) -> PSID {
        self.0
    }
}

impl Drop for LocalSid {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { LocalFree(self.0 as HLOCAL) };
        }
    }
}

struct OwnedHandle(HANDLE);

impl Drop for OwnedHandle {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { CloseHandle(self.0) };
        }
    }
}

unsafe fn set_path_ace(path: &Path, sid: PSID, mode: i32, mask: u32) -> Result<()> {
    let mut security_descriptor = ptr::null_mut();
    let mut old_acl: *mut ACL = ptr::null_mut();
    let path_wide = to_wide(path.as_os_str());
    let status = GetNamedSecurityInfoW(
        path_wide.as_ptr(),
        windows_sys::Win32::Security::Authorization::SE_FILE_OBJECT,
        DACL_SECURITY_INFORMATION,
        ptr::null_mut(),
        ptr::null_mut(),
        &mut old_acl,
        ptr::null_mut(),
        &mut security_descriptor,
    );
    if status != ERROR_SUCCESS {
        return Err(win32_error("GetNamedSecurityInfoW", status));
    }
    let security_descriptor = LocalMemory(security_descriptor as HLOCAL);
    let entry = EXPLICIT_ACCESS_W {
        grfAccessPermissions: mask,
        grfAccessMode: mode,
        grfInheritance: SUB_CONTAINERS_AND_OBJECTS_INHERIT,
        Trustee: TRUSTEE_W {
            pMultipleTrustee: ptr::null_mut(),
            MultipleTrusteeOperation: 0,
            TrusteeForm: TRUSTEE_IS_SID,
            TrusteeType: TRUSTEE_IS_UNKNOWN,
            ptstrName: sid as *mut u16,
        },
    };
    let mut new_acl = ptr::null_mut();
    let status = SetEntriesInAclW(1, &entry, old_acl, &mut new_acl);
    if status != ERROR_SUCCESS {
        return Err(win32_error("SetEntriesInAclW", status));
    }
    let new_acl_memory = LocalMemory(new_acl as HLOCAL);
    let status = SetNamedSecurityInfoW(
        path_wide.as_ptr() as *mut u16,
        windows_sys::Win32::Security::Authorization::SE_FILE_OBJECT,
        DACL_SECURITY_INFORMATION,
        ptr::null_mut(),
        ptr::null_mut(),
        new_acl,
        ptr::null_mut(),
    );
    drop(new_acl_memory);
    drop(security_descriptor);
    if status != ERROR_SUCCESS {
        return Err(win32_error("SetNamedSecurityInfoW", status));
    }
    Ok(())
}

struct LocalMemory(HLOCAL);

impl Drop for LocalMemory {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { LocalFree(self.0) };
        }
    }
}

unsafe fn create_restricted_token(capabilities: &mut [LocalSid]) -> Result<OwnedHandle> {
    let desired = TOKEN_DUPLICATE
        | TOKEN_QUERY
        | TOKEN_ASSIGN_PRIMARY
        | TOKEN_ADJUST_DEFAULT
        | TOKEN_ADJUST_SESSIONID
        | TOKEN_ADJUST_PRIVILEGES;
    let mut base_token = ptr::null_mut();
    if windows_sys::Win32::System::Threading::OpenProcessToken(
        GetCurrentProcess(),
        desired,
        &mut base_token,
    ) == 0
    {
        return Err(last_error("OpenProcessToken"));
    }
    let base_token = OwnedHandle(base_token);
    let mut logon_sid = get_logon_sid(base_token.0)?;
    let mut everyone_sid = world_sid()?;
    let mut entries = capabilities
        .iter_mut()
        .map(|sid| SID_AND_ATTRIBUTES {
            Sid: sid.as_ptr(),
            Attributes: 0,
        })
        .collect::<Vec<_>>();
    entries.push(SID_AND_ATTRIBUTES {
        Sid: logon_sid.as_mut_ptr() as PSID,
        Attributes: 0,
    });
    entries.push(SID_AND_ATTRIBUTES {
        Sid: everyone_sid.as_mut_ptr() as PSID,
        Attributes: 0,
    });
    let mut restricted = ptr::null_mut();
    if CreateRestrictedToken(
        base_token.0,
        DISABLE_MAX_PRIVILEGE | LUA_TOKEN | WRITE_RESTRICTED,
        0,
        ptr::null(),
        0,
        ptr::null(),
        entries.len() as u32,
        entries.as_ptr(),
        &mut restricted,
    ) == 0
    {
        return Err(last_error("CreateRestrictedToken"));
    }
    let restricted = OwnedHandle(restricted);
    set_token_default_dacl(restricted.0, &entries)?;
    enable_traverse_privilege(restricted.0)?;
    Ok(restricted)
}

unsafe fn enable_traverse_privilege(token: HANDLE) -> Result<()> {
    let mut luid = std::mem::zeroed();
    if LookupPrivilegeValueW(
        ptr::null(),
        to_wide("SeChangeNotifyPrivilege").as_ptr(),
        &mut luid,
    ) == 0
    {
        return Err(last_error("LookupPrivilegeValueW(SeChangeNotifyPrivilege)"));
    }
    let mut privileges: TOKEN_PRIVILEGES = std::mem::zeroed();
    privileges.PrivilegeCount = 1;
    privileges.Privileges[0].Luid = luid;
    privileges.Privileges[0].Attributes = 0x0000_0002;
    if AdjustTokenPrivileges(token, 0, &privileges, 0, ptr::null_mut(), ptr::null_mut()) == 0 {
        return Err(last_error("AdjustTokenPrivileges(SeChangeNotifyPrivilege)"));
    }
    Ok(())
}

unsafe fn world_sid() -> Result<Vec<u8>> {
    let mut size = 0;
    CreateWellKnownSid(WIN_WORLD_SID, ptr::null_mut(), ptr::null_mut(), &mut size);
    let mut buffer = vec![0; size as usize];
    if CreateWellKnownSid(
        WIN_WORLD_SID,
        ptr::null_mut(),
        buffer.as_mut_ptr() as PSID,
        &mut size,
    ) == 0
    {
        return Err(last_error("CreateWellKnownSid"));
    }
    Ok(buffer)
}

unsafe fn get_logon_sid(token: HANDLE) -> Result<Vec<u8>> {
    let mut needed = 0;
    GetTokenInformation(token, TokenGroups, ptr::null_mut(), 0, &mut needed);
    let mut buffer = vec![0; needed as usize];
    if needed == 0
        || GetTokenInformation(
            token,
            TokenGroups,
            buffer.as_mut_ptr() as *mut c_void,
            needed,
            &mut needed,
        ) == 0
    {
        return Err(last_error("GetTokenInformation(TokenGroups)"));
    }
    let groups = &*(buffer.as_ptr() as *const TOKEN_GROUPS);
    let first = groups.Groups.as_ptr();
    for index in 0..groups.GroupCount as usize {
        let group = &*first.add(index);
        if group.Attributes & SE_GROUP_LOGON_ID == SE_GROUP_LOGON_ID {
            let length = GetLengthSid(group.Sid);
            let mut sid = vec![0; length as usize];
            if length == 0 || CopySid(length, sid.as_mut_ptr() as PSID, group.Sid) == 0 {
                return Err(last_error("CopySid(logon)"));
            }
            return Ok(sid);
        }
    }
    Err(tool_error("Logon SID is not present on the current token"))
}

#[repr(C)]
struct TokenDefaultDaclInfo {
    default_dacl: *mut ACL,
}

unsafe fn set_token_default_dacl(token: HANDLE, sids: &[SID_AND_ATTRIBUTES]) -> Result<()> {
    let entries = sids
        .iter()
        .map(|sid| EXPLICIT_ACCESS_W {
            grfAccessPermissions: 0x1000_0000,
            grfAccessMode: windows_sys::Win32::Security::Authorization::GRANT_ACCESS,
            grfInheritance: 0,
            Trustee: TRUSTEE_W {
                pMultipleTrustee: ptr::null_mut(),
                MultipleTrusteeOperation: 0,
                TrusteeForm: TRUSTEE_IS_SID,
                TrusteeType: TRUSTEE_IS_UNKNOWN,
                ptstrName: sid.Sid as *mut u16,
            },
        })
        .collect::<Vec<_>>();
    let mut acl = ptr::null_mut();
    let status = SetEntriesInAclW(
        entries.len() as u32,
        entries.as_ptr(),
        ptr::null(),
        &mut acl,
    );
    if status != ERROR_SUCCESS {
        return Err(win32_error("SetEntriesInAclW(default token DACL)", status));
    }
    let acl_memory = LocalMemory(acl as HLOCAL);
    let mut info = TokenDefaultDaclInfo { default_dacl: acl };
    if SetTokenInformation(
        token,
        TokenDefaultDacl,
        &mut info as *mut _ as *mut c_void,
        std::mem::size_of::<TokenDefaultDaclInfo>() as u32,
    ) == 0
    {
        return Err(last_error("SetTokenInformation(TokenDefaultDacl)"));
    }
    drop(acl_memory);
    Ok(())
}

unsafe fn spawn_and_capture(
    prepared: &PreparedCommand,
    token: HANDLE,
    timeout: Duration,
) -> Result<std::process::Output> {
    let mut stdout_read = ptr::null_mut();
    let mut stdout_write = ptr::null_mut();
    let mut stderr_read = ptr::null_mut();
    let mut stderr_write = ptr::null_mut();
    let mut stdin_read = ptr::null_mut();
    let mut stdin_write = ptr::null_mut();
    create_pipe(&mut stdout_read, &mut stdout_write, "stdout")?;
    create_pipe(&mut stderr_read, &mut stderr_write, "stderr")?;
    create_pipe(&mut stdin_read, &mut stdin_write, "stdin")?;
    let stdout_read = OwnedHandle(stdout_read);
    let stdout_write = OwnedHandle(stdout_write);
    let stderr_read = OwnedHandle(stderr_read);
    let stderr_write = OwnedHandle(stderr_write);
    let stdin_read = OwnedHandle(stdin_read);
    let stdin_write = OwnedHandle(stdin_write);
    for handle in [stdout_write.0, stderr_write.0, stdin_read.0] {
        if SetHandleInformation(handle, HANDLE_FLAG_INHERIT, HANDLE_FLAG_INHERIT) == 0 {
            return Err(last_error("SetHandleInformation(child pipe)"));
        }
    }

    let job = create_job()?;
    let mut attrs = AttributeList::new(2)?;
    attrs.set_handles(&[stdin_read.0, stdout_write.0, stderr_write.0])?;
    attrs.set_job(job.0)?;
    let mut startup: STARTUPINFOEXW = std::mem::zeroed();
    startup.StartupInfo.cb = std::mem::size_of::<STARTUPINFOEXW>() as u32;
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    startup.StartupInfo.hStdInput = stdin_read.0;
    startup.StartupInfo.hStdOutput = stdout_write.0;
    startup.StartupInfo.hStdError = stderr_write.0;
    let mut desktop = to_wide("winsta0\\default");
    startup.StartupInfo.lpDesktop = desktop.as_mut_ptr();
    startup.lpAttributeList = attrs.as_mut_ptr();

    let mut argv = vec![prepared.program.clone()];
    argv.extend(prepared.args.clone());
    let mut command_line = to_wide(argv_to_command_line(&argv));
    let cwd = prepared.cwd.as_deref().unwrap_or_else(|| Path::new("."));
    let cwd_wide = to_wide(cwd.as_os_str());
    let env_block = environment_block(&prepared.env);
    let mut process: PROCESS_INFORMATION = std::mem::zeroed();
    if CreateProcessAsUserW(
        token,
        ptr::null(),
        command_line.as_mut_ptr(),
        ptr::null(),
        ptr::null(),
        1,
        CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT | CREATE_NO_WINDOW,
        env_block.as_ptr() as *const c_void,
        cwd_wide.as_ptr(),
        &startup.StartupInfo,
        &mut process,
    ) == 0
    {
        return Err(last_error("CreateProcessAsUserW"));
    }
    let process_handle = OwnedHandle(process.hProcess);
    let thread_handle = OwnedHandle(process.hThread);
    drop(stdin_read);
    drop(stdin_write);
    drop(stdout_write);
    drop(stderr_write);

    let stdout_thread = read_pipe(stdout_read);
    let stderr_thread = read_pipe(stderr_read);
    let timeout_ms = timeout.as_millis().min(u32::MAX as u128) as u32;
    let wait = WaitForSingleObject(process_handle.0, timeout_ms);
    let timed_out = wait == WAIT_TIMEOUT;
    if timed_out {
        TerminateJobObject(job.0, 1);
        WaitForSingleObject(process_handle.0, INFINITE);
    } else if wait != WAIT_OBJECT_0 {
        return Err(last_error("WaitForSingleObject"));
    }
    let mut exit_code = 1;
    if GetExitCodeProcess(process_handle.0, &mut exit_code) == 0 {
        return Err(last_error("GetExitCodeProcess"));
    }
    drop(thread_handle);
    drop(process_handle);
    drop(job);
    let stdout = stdout_thread
        .join()
        .map_err(|_| tool_error("stdout reader thread panicked"))?;
    let stderr = stderr_thread
        .join()
        .map_err(|_| tool_error("stderr reader thread panicked"))?;
    if timed_out {
        return Err(tool_error(&format!(
            "Command timed out after {} seconds",
            timeout.as_secs_f64()
        )));
    }
    Ok(std::process::Output {
        status: std::process::ExitStatus::from_raw(exit_code),
        stdout,
        stderr,
    })
}

unsafe fn create_pipe(read: &mut HANDLE, write: &mut HANDLE, name: &str) -> Result<()> {
    if CreatePipe(read, write, ptr::null(), 0) == 0 {
        return Err(last_error(&format!("CreatePipe({name})")));
    }
    Ok(())
}

fn read_pipe(handle: OwnedHandle) -> std::thread::JoinHandle<Vec<u8>> {
    let raw = handle.0 as usize;
    std::mem::forget(handle);
    std::thread::spawn(move || {
        let handle = OwnedHandle(raw as HANDLE);
        let mut output = Vec::new();
        let mut buffer = [0; 8192];
        loop {
            let mut read = 0;
            let ok = unsafe {
                ReadFile(
                    handle.0,
                    buffer.as_mut_ptr(),
                    buffer.len() as u32,
                    &mut read,
                    ptr::null_mut(),
                )
            };
            if ok == 0 || read == 0 {
                break;
            }
            output.extend_from_slice(&buffer[..read as usize]);
        }
        output
    })
}

unsafe fn create_job() -> Result<OwnedHandle> {
    let job = OwnedHandle(CreateJobObjectW(ptr::null(), ptr::null()));
    if job.0.is_null() {
        return Err(last_error("CreateJobObjectW"));
    }
    let mut limits: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
    limits.BasicLimitInformation.LimitFlags =
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_BREAKAWAY_OK;
    if SetInformationJobObject(
        job.0,
        JobObjectExtendedLimitInformation,
        &limits as *const _ as *const c_void,
        std::mem::size_of_val(&limits) as u32,
    ) == 0
    {
        return Err(last_error("SetInformationJobObject"));
    }
    Ok(job)
}

struct AttributeList {
    buffer: Vec<u8>,
    handles: Vec<HANDLE>,
    jobs: Vec<HANDLE>,
}

impl AttributeList {
    unsafe fn new(count: u32) -> Result<Self> {
        let mut size = 0;
        InitializeProcThreadAttributeList(ptr::null_mut(), count, 0, &mut size);
        if size == 0 {
            return Err(last_error("InitializeProcThreadAttributeList(size)"));
        }
        let mut this = Self {
            buffer: vec![0; size],
            handles: Vec::new(),
            jobs: Vec::new(),
        };
        if InitializeProcThreadAttributeList(this.as_mut_ptr(), count, 0, &mut size) == 0 {
            return Err(last_error("InitializeProcThreadAttributeList"));
        }
        Ok(this)
    }

    fn as_mut_ptr(&mut self) -> LPPROC_THREAD_ATTRIBUTE_LIST {
        self.buffer.as_mut_ptr() as LPPROC_THREAD_ATTRIBUTE_LIST
    }

    unsafe fn set_handles(&mut self, handles: &[HANDLE]) -> Result<()> {
        self.handles = handles.to_vec();
        self.update(
            PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            self.handles.as_ptr() as *const c_void,
            std::mem::size_of_val(self.handles.as_slice()),
        )
    }

    unsafe fn set_job(&mut self, job: HANDLE) -> Result<()> {
        self.jobs = vec![job];
        self.update(
            PROC_THREAD_ATTRIBUTE_JOB_LIST,
            self.jobs.as_ptr() as *const c_void,
            std::mem::size_of_val(self.jobs.as_slice()),
        )
    }

    unsafe fn update(&mut self, attribute: usize, value: *const c_void, size: usize) -> Result<()> {
        if UpdateProcThreadAttribute(
            self.as_mut_ptr(),
            0,
            attribute,
            value,
            size,
            ptr::null_mut(),
            ptr::null(),
        ) == 0
        {
            return Err(last_error("UpdateProcThreadAttribute"));
        }
        Ok(())
    }
}

impl Drop for AttributeList {
    fn drop(&mut self) {
        unsafe { DeleteProcThreadAttributeList(self.as_mut_ptr()) };
    }
}

fn environment_block(overrides: &BTreeMap<String, String>) -> Vec<u16> {
    let mut variables = std::env::vars().collect::<BTreeMap<_, _>>();
    for (key, value) in overrides {
        if let Some(existing) = variables
            .keys()
            .find(|candidate| candidate.eq_ignore_ascii_case(key))
            .cloned()
        {
            variables.remove(&existing);
        }
        variables.insert(key.clone(), value.clone());
    }
    let mut block = Vec::new();
    for (key, value) in variables {
        block.extend(OsStr::new(&format!("{key}={value}")).encode_wide());
        block.push(0);
    }
    block.push(0);
    block
}

fn argv_to_command_line(args: &[String]) -> String {
    args.iter()
        .map(|arg| quote_arg(arg))
        .collect::<Vec<_>>()
        .join(" ")
}

fn quote_arg(arg: &str) -> String {
    if !arg.is_empty()
        && !arg
            .chars()
            .any(|character| matches!(character, ' ' | '\t' | '\n' | '\r' | '"'))
    {
        return arg.to_string();
    }
    let mut quoted = String::from("\"");
    let mut backslashes = 0;
    for character in arg.chars() {
        match character {
            '\\' => backslashes += 1,
            '"' => {
                quoted.push_str(&"\\".repeat(backslashes * 2 + 1));
                quoted.push('"');
                backslashes = 0;
            }
            _ => {
                quoted.push_str(&"\\".repeat(backslashes));
                backslashes = 0;
                quoted.push(character);
            }
        }
    }
    quoted.push_str(&"\\".repeat(backslashes * 2));
    quoted.push('"');
    quoted
}

fn to_wide(value: impl AsRef<OsStr>) -> Vec<u16> {
    value.as_ref().encode_wide().chain(Some(0)).collect()
}

fn last_error(operation: &str) -> DeepCodeError {
    let code = unsafe { GetLastError() };
    tool_error(&format!("{operation} failed with Windows error {code}"))
}

fn win32_error(operation: &str, code: u32) -> DeepCodeError {
    tool_error(&format!("{operation} failed with Windows error {code}"))
}

fn tool_error(message: &str) -> DeepCodeError {
    DeepCodeError::ToolExecution {
        tool: "sandbox".into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_sids_are_stable_and_path_scoped() {
        assert_eq!(capability_sid("c:/one"), capability_sid("c:/one"));
        assert_ne!(capability_sid("c:/one"), capability_sid("c:/two"));
    }

    #[test]
    fn quoting_preserves_spaces_quotes_and_trailing_slashes() {
        assert_eq!(quote_arg("plain"), "plain");
        assert_eq!(quote_arg("two words"), "\"two words\"");
        assert_eq!(quote_arg("a\\\"b"), "\"a\\\\\\\"b\"");
        assert_eq!(quote_arg("a b\\"), "\"a b\\\\\"");
    }
}
