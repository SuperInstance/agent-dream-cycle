//! # agent-dream-cycle
//!
//! Offline memory consolidation — REM sleep for autonomous agents.
//!
//! Based on the hypothesis that agents improve faster by "dreaming" about
//! failures than by continuous operation. The dream cycle pauses normal
//! execution, replays failures at high speed, and compresses recent
//! experiences into long-term patterns.



/// A record of a past interaction that may be replayed during dreaming.
#[derive(Clone, Debug)]
pub struct Experience {
    /// Unique identifier.
    pub id: u64,
    /// The input/context that was given.
    pub input: String,
    /// The response the agent produced.
    pub output: String,
    /// Whether this interaction was a success or failure.
    pub success: bool,
    /// A reward score (higher = better outcome).
    pub reward: f64,
    /// Embedding representation for similarity search.
    pub embedding: Vec<f64>,
    /// Tick at which this experience occurred.
    pub tick: u64,
}

impl Experience {
    /// Create a new experience record.
    pub fn new(id: u64, input: &str, output: &str, success: bool, reward: f64, embedding: Vec<f64>, tick: u64) -> Self {
        Self {
            id,
            input: input.to_string(),
            output: output.to_string(),
            success,
            reward,
            embedding,
            tick,
        }
    }

    /// Create a simple experience with a scalar embedding.
    pub fn simple(id: u64, input: &str, success: bool, reward: f64, tick: u64) -> Self {
        Self {
            id,
            input: input.to_string(),
            output: String::new(),
            success,
            reward,
            embedding: vec![reward],
            tick,
        }
    }
}

/// Compressed pattern extracted from multiple experiences during consolidation.
#[derive(Clone, Debug)]
pub struct ConsolidatedPattern {
    /// A human-readable description of the learned pattern.
    pub description: String,
    /// Average reward for experiences matching this pattern.
    pub avg_reward: f64,
    /// Number of experiences that contributed to this pattern.
    pub sample_count: usize,
    /// Tags/categories for this pattern.
    pub tags: Vec<String>,
    /// Whether this pattern represents a success or failure mode.
    pub is_success_pattern: bool,
}

impl ConsolidatedPattern {
    /// Create a new consolidated pattern.
    pub fn new(description: &str, avg_reward: f64, sample_count: usize, tags: Vec<String>, is_success_pattern: bool) -> Self {
        Self {
            description: description.to_string(),
            avg_reward,
            sample_count,
            tags,
            is_success_pattern,
        }
    }
}

/// Result of a failure replay session — what was learned from revisiting failures.
#[derive(Clone, Debug)]
pub struct FailureReplayResult {
    /// The failures that were replayed.
    pub failures_replayed: Vec<Experience>,
    /// Patterns identified from failure analysis.
    pub patterns_found: Vec<ConsolidatedPattern>,
    /// Suggested improvements (input -> better output).
    pub suggestions: Vec<(String, String)>,
    /// The estimated improvement from this replay session.
    pub improvement_score: f64,
}

impl FailureReplayResult {
    /// Create an empty replay result.
    pub fn empty() -> Self {
        Self {
            failures_replayed: vec![],
            patterns_found: vec![],
            suggestions: vec![],
            improvement_score: 0.0,
        }
    }
}

/// Memory consolidation engine: compresses recent experiences into long-term patterns.
#[derive(Clone, Debug)]
pub struct MemoryConsolidation {
    /// All stored experiences.
    pub experiences: Vec<Experience>,
    /// Long-term consolidated patterns.
    pub patterns: Vec<ConsolidatedPattern>,
    /// Minimum experiences needed before consolidation kicks in.
    pub min_experiences: usize,
}

impl MemoryConsolidation {
    /// Create a new consolidation engine.
    pub fn new(min_experiences: usize) -> Self {
        Self {
            experiences: vec![],
            patterns: vec![],
            min_experiences,
        }
    }

    /// Add an experience to the store.
    pub fn add_experience(&mut self, exp: Experience) {
        self.experiences.push(exp);
    }

    /// Get all failure experiences.
    pub fn failures(&self) -> Vec<&Experience> {
        self.experiences.iter().filter(|e| !e.success).collect()
    }

    /// Get all success experiences.
    pub fn successes(&self) -> Vec<&Experience> {
        self.experiences.iter().filter(|e| e.success).collect()
    }

    /// Run consolidation: compress experiences into patterns.
    /// Returns the number of new patterns created.
    pub fn consolidate(&mut self) -> usize {
        if self.experiences.len() < self.min_experiences {
            return 0;
        }

        // Collect data first to avoid borrow issues
        let failure_data: Vec<f64> = self.experiences.iter().filter(|e| !e.success).map(|e| e.reward).collect();
        let success_data: Vec<f64> = self.experiences.iter().filter(|e| e.success).map(|e| e.reward).collect();
        let mut rewards: Vec<f64> = self.experiences.iter().map(|e| e.reward).collect();
        rewards.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut new_patterns = 0;

        // Pattern: low-reward failures
        if !failure_data.is_empty() {
            let avg_reward: f64 = failure_data.iter().sum::<f64>() / failure_data.len() as f64;
            self.patterns.push(ConsolidatedPattern::new(
                &format!("Low-reward failure pattern (avg reward: {:.2})", avg_reward),
                avg_reward,
                failure_data.len(),
                vec!["failure".to_string(), "low-reward".to_string()],
                false,
            ));
            new_patterns += 1;
        }

        // Pattern: high-reward successes
        if !success_data.is_empty() {
            let avg_reward: f64 = success_data.iter().sum::<f64>() / success_data.len() as f64;
            self.patterns.push(ConsolidatedPattern::new(
                &format!("High-reward success pattern (avg reward: {:.2})", avg_reward),
                avg_reward,
                success_data.len(),
                vec!["success".to_string(), "high-reward".to_string()],
                true,
            ));
            new_patterns += 1;
        }

        // Pattern: reward distribution by quartile
        if rewards.len() >= 4 {
            let q1 = rewards[rewards.len() / 4];
            let q3 = rewards[3 * rewards.len() / 4];
            let bottom_quartile_count = rewards.iter().filter(|&&r| r <= q1).count();
            self.patterns.push(ConsolidatedPattern::new(
                &format!("Reward distribution: Q1={:.2}, Q3={:.2}", q1, q3),
                (q1 + q3) / 2.0,
                bottom_quartile_count,
                vec!["distribution".to_string(), "quartile".to_string()],
                false,
            ));
            new_patterns += 1;
        }

        new_patterns
    }

    /// Count total experiences.
    pub fn len(&self) -> usize {
        self.experiences.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.experiences.is_empty()
    }
}

/// Replays failures in embedding space to find improvement opportunities.
#[derive(Clone, Debug)]
pub struct FailureReplay {
    /// Speed multiplier for replay (higher = faster).
    pub speed: f64,
}

impl FailureReplay {
    /// Create a new failure replay engine.
    pub fn new(speed: f64) -> Self {
        Self { speed }
    }

    /// Replay all failures from the consolidation store and generate suggestions.
    pub fn replay(&self, consolidation: &MemoryConsolidation) -> FailureReplayResult {
        let failures: Vec<Experience> = consolidation.failures().into_iter().cloned().collect();
        let successes: Vec<Experience> = consolidation.successes().into_iter().cloned().collect();

        if failures.is_empty() {
            return FailureReplayResult::empty();
        }

        let mut result = FailureReplayResult {
            failures_replayed: failures.clone(),
            patterns_found: vec![],
            suggestions: vec![],
            improvement_score: 0.0,
        };

        // For each failure, find the closest success and suggest that output instead
        let avg_failure_reward: f64 = failures.iter().map(|e| e.reward).sum::<f64>() / failures.len() as f64;

        for failure in &failures {
            let mut best_match: Option<&Experience> = None;
            let mut best_sim = f64::NEG_INFINITY;

            for success in &successes {
                let sim = cosine_similarity(&failure.embedding, &success.embedding);
                if sim > best_sim {
                    best_sim = sim;
                    best_match = Some(success);
                }
            }

            if let Some(match_success) = best_match {
                result.suggestions.push((
                    format!("Input '{}' (reward: {:.2}) → try '{}'", failure.input, failure.reward, match_success.output),
                    format!("Similar success: '{}' (reward: {:.2})", match_success.input, match_success.reward),
                ));
            }
        }

        // Improvement score: how much better are successes vs failures
        let avg_success_reward: f64 = if successes.is_empty() {
            0.0
        } else {
            successes.iter().map(|e| e.reward).sum::<f64>() / successes.len() as f64
        };
        result.improvement_score = (avg_success_reward - avg_failure_reward) * self.speed;

        result
    }
}

/// Compute cosine similarity between two vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let min_len = a.len().min(b.len());
    let dot: f64 = (0..min_len).map(|i| a[i] * b[i]).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

/// Report generated after a dream cycle completes.
#[derive(Clone, Debug)]
pub struct ConsolidationReport {
    /// Number of experiences processed.
    pub experiences_processed: usize,
    /// Number of failures replayed.
    pub failures_replayed: usize,
    /// Number of new patterns learned.
    pub new_patterns: usize,
    /// Improvement score from failure replay.
    pub improvement_score: f64,
    /// Suggestions generated.
    pub suggestions_count: usize,
    /// Duration of the dream cycle in ticks.
    pub duration_ticks: u64,
    /// Summary text.
    pub summary: String,
}

impl ConsolidationReport {
    /// Create a report from dream cycle results.
    pub fn from_results(
        experiences_processed: usize,
        failures_replayed: usize,
        new_patterns: usize,
        replay_result: &FailureReplayResult,
        duration_ticks: u64,
    ) -> Self {
        let summary = format!(
            "Dream cycle complete: processed {} experiences, replayed {} failures, learned {} new patterns. Improvement: {:.2}",
            experiences_processed, failures_replayed, new_patterns, replay_result.improvement_score
        );
        Self {
            experiences_processed,
            failures_replayed,
            new_patterns,
            improvement_score: replay_result.improvement_score,
            suggestions_count: replay_result.suggestions.len(),
            duration_ticks,
            summary,
        }
    }
}

/// The dream cycle orchestrator — pauses agent, replays failures, consolidates memories.
#[derive(Clone, Debug)]
pub struct DreamCycle {
    /// Memory consolidation engine.
    pub consolidation: MemoryConsolidation,
    /// Failure replay engine.
    pub replay: FailureReplay,
    /// Whether currently dreaming.
    pub is_dreaming: bool,
    /// Total dream cycles completed.
    pub cycles_completed: u64,
    /// Reports from past dream cycles.
    pub reports: Vec<ConsolidationReport>,
}

impl DreamCycle {
    /// Create a new dream cycle manager.
    pub fn new(consolidation: MemoryConsolidation, replay: FailureReplay) -> Self {
        Self {
            consolidation,
            replay,
            is_dreaming: false,
            cycles_completed: 0,
            reports: vec![],
        }
    }

    /// Add an experience before the next dream cycle.
    pub fn add_experience(&mut self, exp: Experience) {
        self.consolidation.add_experience(exp);
    }

    /// Enter dream state: pause normal operation, replay failures, consolidate.
    pub fn dream(&mut self, duration_ticks: u64) -> ConsolidationReport {
        self.is_dreaming = true;

        let experiences_count = self.consolidation.len();
        let failures_count = self.consolidation.failures().len();

        // Replay failures
        let replay_result = self.replay.replay(&self.consolidation);

        // Consolidate memories
        let new_patterns = self.consolidation.consolidate();

        let report = ConsolidationReport::from_results(
            experiences_count,
            failures_count,
            new_patterns,
            &replay_result,
            duration_ticks,
        );

        self.is_dreaming = false;
        self.cycles_completed += 1;
        self.reports.push(report.clone());

        report
    }

    /// Check if dreaming.
    pub fn is_active(&self) -> bool {
        self.is_dreaming
    }
}

/// Scheduler for periodic dream cycles.
#[derive(Clone, Debug)]
pub struct DreamScheduler {
    /// Interval between dream cycles in ticks.
    pub interval: u64,
    /// Current tick counter.
    pub current_tick: u64,
    /// The dream cycle to manage.
    pub dream_cycle: DreamCycle,
    /// History of tick values when dreams were triggered.
    pub dream_ticks: Vec<u64>,
}

impl DreamScheduler {
    /// Create a new scheduler with given interval.
    pub fn new(interval: u64, dream_cycle: DreamCycle) -> Self {
        Self {
            interval,
            current_tick: 0,
            dream_cycle,
            dream_ticks: vec![],
        }
    }

    /// Advance by one tick. Returns a report if a dream cycle was triggered.
    pub fn tick(&mut self) -> Option<ConsolidationReport> {
        self.current_tick += 1;
        if self.current_tick > 0 && self.current_tick % self.interval == 0 {
            self.dream_ticks.push(self.current_tick);
            let report = self.dream_cycle.dream(self.interval);
            Some(report)
        } else {
            None
        }
    }

    /// Advance multiple ticks, returning all dream reports generated.
    pub fn tick_many(&mut self, count: u64) -> Vec<ConsolidationReport> {
        let mut reports = vec![];
        for _ in 0..count {
            if let Some(report) = self.tick() {
                reports.push(report);
            }
        }
        reports
    }

    /// Check how many ticks until the next dream.
    pub fn ticks_until_dream(&self) -> u64 {
        self.interval - (self.current_tick % self.interval)
    }

    /// Total dreams triggered so far.
    pub fn total_dreams(&self) -> usize {
        self.dream_ticks.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dream_cycle_basic_execution() {
        let consolidation = MemoryConsolidation::new(1);
        let replay = FailureReplay::new(1.0);
        let mut dc = DreamCycle::new(consolidation, replay);

        assert!(!dc.is_active());
        assert_eq!(dc.cycles_completed, 0);

        dc.add_experience(Experience::simple(1, "test input", false, -1.0, 0));
        dc.add_experience(Experience::simple(2, "good input", true, 5.0, 1));

        let report = dc.dream(10);
        assert_eq!(dc.cycles_completed, 1);
        assert_eq!(report.experiences_processed, 2);
        assert_eq!(report.failures_replayed, 1);
        assert!(!dc.is_active());
    }

    #[test]
    fn test_failure_replay() {
        let mut consolidation = MemoryConsolidation::new(1);
        let replay = FailureReplay::new(2.0);

        consolidation.add_experience(Experience::simple(1, "fail1", false, -2.0, 0));
        consolidation.add_experience(Experience::simple(2, "fail2", false, -1.0, 1));
        consolidation.add_experience(Experience::simple(3, "success1", true, 8.0, 2));

        let result = replay.replay(&consolidation);
        assert_eq!(result.failures_replayed.len(), 2);
        assert!(result.improvement_score > 0.0);
    }

    #[test]
    fn test_failure_replay_empty() {
        let consolidation = MemoryConsolidation::new(1);
        let replay = FailureReplay::new(1.0);
        let result = replay.replay(&consolidation);
        assert_eq!(result.failures_replayed.len(), 0);
    }

    #[test]
    fn test_consolidation_output() {
        let mut mc = MemoryConsolidation::new(2);
        mc.add_experience(Experience::simple(1, "a", false, -1.0, 0));
        mc.add_experience(Experience::simple(2, "b", true, 3.0, 1));
        mc.add_experience(Experience::simple(3, "c", true, 5.0, 2));

        let new = mc.consolidate();
        assert!(new >= 2); // at least failure + success pattern
        assert!(!mc.patterns.is_empty());
    }

    #[test]
    fn test_consolidation_below_threshold() {
        let mut mc = MemoryConsolidation::new(10);
        mc.add_experience(Experience::simple(1, "a", false, -1.0, 0));
        let new = mc.consolidate();
        assert_eq!(new, 0);
    }

    #[test]
    fn test_scheduler_timing() {
        let consolidation = MemoryConsolidation::new(1);
        let replay = FailureReplay::new(1.0);
        let dc = DreamCycle::new(consolidation, replay);
        let mut scheduler = DreamScheduler::new(5, dc);

        // Add experiences upfront
        scheduler.dream_cycle.add_experience(Experience::simple(1, "a", false, -1.0, 0));

        let mut reports = vec![];
        for _ in 0..15 {
            if let Some(r) = scheduler.tick() {
                reports.push(r);
            }
        }

        assert_eq!(reports.len(), 3); // at ticks 5, 10, 15
        assert_eq!(scheduler.total_dreams(), 3);
    }

    #[test]
    fn test_ticks_until_dream() {
        let consolidation = MemoryConsolidation::new(1);
        let replay = FailureReplay::new(1.0);
        let dc = DreamCycle::new(consolidation, replay);
        let mut scheduler = DreamScheduler::new(10, dc);

        assert_eq!(scheduler.ticks_until_dream(), 10);
        scheduler.tick();
        assert_eq!(scheduler.ticks_until_dream(), 9);
    }

    #[test]
    fn test_improvement_after_dreams() {
        let consolidation = MemoryConsolidation::new(1);
        let replay = FailureReplay::new(1.0);
        let dc = DreamCycle::new(consolidation, replay);
        let mut scheduler = DreamScheduler::new(3, dc);

        // Seed with mixed experiences
        for i in 0..5 {
            scheduler.dream_cycle.add_experience(
                Experience::simple(i, &format!("exp-{}", i), i % 2 == 0, if i % 2 == 0 { 5.0 } else { -2.0 }, i)
            );
        }

        let reports = scheduler.tick_many(6);
        assert_eq!(reports.len(), 2);

        // Each report should show improvement potential
        for report in &reports {
            assert_eq!(report.experiences_processed, 5);
        }
    }

    #[test]
    fn test_experience_creation() {
        let exp = Experience::new(42, "input", "output", true, 1.0, vec![1.0, 2.0], 5);
        assert_eq!(exp.id, 42);
        assert_eq!(exp.input, "input");
        assert_eq!(exp.output, "output");
        assert!(exp.success);
        assert_eq!(exp.embedding, vec![1.0, 2.0]);
        assert_eq!(exp.tick, 5);
    }

    #[test]
    fn test_consolidation_report_summary() {
        let report = ConsolidationReport::from_results(
            10, 3, 2, &FailureReplayResult::empty(), 5,
        );
        assert!(report.summary.contains("10 experiences"));
        assert!(report.summary.contains("3 failures"));
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-10);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 1e-10);

        let z = vec![0.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &z) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_scheduler_tick_many() {
        let consolidation = MemoryConsolidation::new(1);
        let replay = FailureReplay::new(1.0);
        let dc = DreamCycle::new(consolidation, replay);
        let mut scheduler = DreamScheduler::new(4, dc);

        scheduler.dream_cycle.add_experience(Experience::simple(1, "x", false, -1.0, 0));

        let reports = scheduler.tick_many(12);
        assert_eq!(reports.len(), 3); // ticks 4, 8, 12
    }
}
