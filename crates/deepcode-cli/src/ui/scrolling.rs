//! Scroll state tracking for the alternate-screen transcript.

use std::time::{Duration, Instant};

const TAIL_SENTINEL: usize = usize::MAX;
const TRACKPAD_EVENT_WINDOW: Duration = Duration::from_millis(35);
const WHEEL_LINES_PER_TICK: i32 = 3;
const TRACKPAD_BASE_LINES_PER_TICK: i32 = 1;
const TRACKPAD_MID_LINES_PER_TICK: i32 = 2;
const TRACKPAD_MAX_LINES_PER_TICK: i32 = 3;

/// Flat line-offset scroll state for the transcript view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TranscriptScroll {
    offset: usize,
}

impl Default for TranscriptScroll {
    fn default() -> Self {
        Self::to_bottom()
    }
}

impl TranscriptScroll {
    pub(crate) const fn to_bottom() -> Self {
        Self {
            offset: TAIL_SENTINEL,
        }
    }

    pub(crate) const fn at_line(offset: usize) -> Self {
        Self { offset }
    }

    pub(crate) fn resolve_top(self, max_start: usize) -> (Self, usize) {
        if self.offset == TAIL_SENTINEL {
            return (Self::to_bottom(), max_start);
        }

        let top = self.offset.min(max_start);
        if top >= max_start {
            (Self::to_bottom(), max_start)
        } else {
            (Self::at_line(top), top)
        }
    }

    pub(crate) fn scrolled_by(
        self,
        delta_lines: i32,
        total_lines: usize,
        visible_lines: usize,
    ) -> Self {
        if delta_lines == 0 {
            return self;
        }
        if total_lines <= visible_lines {
            return Self::to_bottom();
        }

        let max_start = total_lines.saturating_sub(visible_lines);
        let current_top = if self.offset == TAIL_SENTINEL {
            max_start
        } else {
            self.offset.min(max_start)
        };

        let new_top = if delta_lines < 0 {
            current_top.saturating_sub(delta_lines.unsigned_abs() as usize)
        } else {
            current_top
                .saturating_add(delta_lines as usize)
                .min(max_start)
        };

        if new_top >= max_start {
            Self::to_bottom()
        } else {
            Self::at_line(new_top)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScrollDirection {
    Up,
    Down,
}

impl ScrollDirection {
    fn sign(self) -> i32 {
        match self {
            Self::Up => -1,
            Self::Down => 1,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct MouseScrollState {
    last_event_at: Option<Instant>,
    last_direction: Option<ScrollDirection>,
    rapid_same_direction_ticks: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScrollUpdate {
    pub delta_lines: i32,
}

impl MouseScrollState {
    pub(crate) fn on_scroll(&mut self, direction: ScrollDirection) -> ScrollUpdate {
        self.on_scroll_at(direction, Instant::now())
    }

    fn on_scroll_at(&mut self, direction: ScrollDirection, now: Instant) -> ScrollUpdate {
        let is_trackpad = self
            .last_event_at
            .is_some_and(|last| now.saturating_duration_since(last) < TRACKPAD_EVENT_WINDOW);
        let same_direction = self.last_direction == Some(direction);

        self.last_event_at = Some(now);
        self.last_direction = Some(direction);

        let lines_per_tick = if is_trackpad {
            if same_direction {
                self.rapid_same_direction_ticks = self.rapid_same_direction_ticks.saturating_add(1);
            } else {
                self.rapid_same_direction_ticks = 1;
            }
            match self.rapid_same_direction_ticks {
                0..=2 => TRACKPAD_BASE_LINES_PER_TICK,
                3..=5 => TRACKPAD_MID_LINES_PER_TICK,
                _ => TRACKPAD_MAX_LINES_PER_TICK,
            }
        } else {
            self.rapid_same_direction_ticks = 0;
            WHEEL_LINES_PER_TICK
        };

        ScrollUpdate {
            delta_lines: direction.sign() * lines_per_tick,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_resolves_to_current_bottom() {
        let (state, top) = TranscriptScroll::to_bottom().resolve_top(42);

        assert_eq!(state, TranscriptScroll::to_bottom());
        assert_eq!(top, 42);
    }

    #[test]
    fn explicit_line_clamps_without_tail_until_bottom() {
        let (state, top) = TranscriptScroll::at_line(9).resolve_top(20);

        assert_eq!(state, TranscriptScroll::at_line(9));
        assert_eq!(top, 9);

        let (state, top) = TranscriptScroll::at_line(99).resolve_top(20);
        assert_eq!(state, TranscriptScroll::to_bottom());
        assert_eq!(top, 20);
    }

    #[test]
    fn scroll_from_tail_moves_up_and_back_to_tail() {
        let up = TranscriptScroll::to_bottom().scrolled_by(-3, 100, 20);

        assert_eq!(up, TranscriptScroll::at_line(77));
        assert_eq!(up.scrolled_by(10, 100, 20), TranscriptScroll::to_bottom());
    }

    #[test]
    fn mouse_wheel_tick_moves_three_lines() {
        let mut state = MouseScrollState::default();

        assert_eq!(
            state.on_scroll_at(ScrollDirection::Down, Instant::now()),
            ScrollUpdate { delta_lines: 3 }
        );
    }

    #[test]
    fn rapid_same_direction_trackpad_accelerates_but_caps() {
        let mut state = MouseScrollState::default();
        let start = Instant::now();

        assert_eq!(
            state.on_scroll_at(ScrollDirection::Down, start).delta_lines,
            3
        );
        assert_eq!(
            state
                .on_scroll_at(ScrollDirection::Down, start + Duration::from_millis(10))
                .delta_lines,
            1
        );
        assert_eq!(
            state
                .on_scroll_at(ScrollDirection::Down, start + Duration::from_millis(20))
                .delta_lines,
            1
        );
        assert_eq!(
            state
                .on_scroll_at(ScrollDirection::Down, start + Duration::from_millis(30))
                .delta_lines,
            2
        );
        assert_eq!(
            state
                .on_scroll_at(ScrollDirection::Down, start + Duration::from_millis(40))
                .delta_lines,
            2
        );
        assert_eq!(
            state
                .on_scroll_at(ScrollDirection::Down, start + Duration::from_millis(50))
                .delta_lines,
            2
        );
        assert_eq!(
            state
                .on_scroll_at(ScrollDirection::Down, start + Duration::from_millis(60))
                .delta_lines,
            3
        );
    }
}
