# agent-dream-cycle

**Offline memory consolidation — REM sleep for autonomous agents.**

An experimental Rust library implementing the hypothesis that agents improve faster by periodically "dreaming" about their failures than by continuous operation. Just as human sleep consolidates memories and replays difficult experiences, this crate provides a framework for agents to pause, replay failures at high speed, and compress recent experiences into long-term patterns.

## Origin

Based on Qwen's insight: *".nail file is a sleeping brain, TelemetryDaemon is REM sleep"*

This crate tests whether structured offline processing — dream cycles — produces measurable improvement in agent performance compared to continuous operation. The analogy to sleep cycles is deliberate: REM sleep appears to replay and integrate difficult experiences, and the same mechanism may benefit autonomous agents.

## Core Concepts

### Experience
The fundamental unit of memory: a record of a past interaction including input, output, success/failure status, reward score, and an embedding representation. Experiences accumulate during normal operation and are processed during dream cycles.

```rust
use agent_dream_cycle::Experience;

let exp = Experience::simple(1, "user query about Rust lifetimes", false, -2.0, 42);
```

### DreamCycle
The orchestrator that manages the dream state. When triggered, it:
1. Pauses normal agent operation
2. Replays all failure experiences through the `FailureReplay` engine
3. Consolidates experiences into long-term patterns via `MemoryConsolidation`
4. Produces a `ConsolidationReport` summarizing what was learned

```rust
use agent_dream_cycle::{DreamCycle, MemoryConsolidation, FailureReplay};

let consolidation = MemoryConsolidation::new(3);
let replay = FailureReplay::new(2.0);
let mut dream = DreamCycle::new(consolidation, replay);

dream.add_experience(Experience::simple(1, "query", false, -1.0, 0));
dream.add_experience(Experience::simple(2, "query", true, 5.0, 1));

let report = dream.dream(10);
println!("{}", report.summary);
```

### MemoryConsolidation
Compresses raw experiences into abstract patterns. It identifies failure patterns (low-reward clusters), success patterns (high-reward clusters), and reward distribution patterns. Requires a minimum number of experiences before consolidation triggers — dreaming with too little data is wasteful.

### FailureReplay
The REM sleep engine: replays failure experiences and matches them against the closest successes. For each failure, it identifies the most similar success (by embedding cosine similarity) and generates a suggestion: "Instead of what you did, try what worked in this similar situation." The improvement score measures the gap between average success and failure rewards.

```rust
use agent_dream_cycle::{FailureReplay, MemoryConsolidation, Experience};

let replay = FailureReplay::new(2.0); // 2x speed multiplier
let result = replay.replay(&consolidation);
println!("Improvement potential: {}", result.improvement_score);
```

### DreamScheduler
Triggers dream cycles at regular intervals (every N ticks). The scheduler tracks time and automatically invokes dream cycles when the interval elapses, returning consolidation reports.

```rust
use agent_dream_cycle::{DreamScheduler, DreamCycle, MemoryConsolidation, FailureReplay, Experience};

let dc = DreamCycle::new(MemoryConsolidation::new(1), FailureReplay::new(1.0));
let mut scheduler = DreamScheduler::new(100, dc); // dream every 100 ticks

// During operation, add experiences
scheduler.dream_cycle.add_experience(Experience::simple(1, "query", false, -3.0, 0));

// Advance time
let reports = scheduler.tick_many(200);
// One dream at tick 100
```

## Design Principles

1. **Failures are teachers.** The dream cycle focuses on failures, not successes. Replaying failures and finding similar successes is more valuable than rehearsing what already worked.

2. **Sleep is productive.** Dream cycles aren't idle time — they're active processing. The `improvement_score` quantifies how much better the agent could perform based on what was learned.

3. **Consolidation compresses.** Raw experiences are verbose; consolidated patterns are compact. The ratio of experiences to patterns measures compression efficiency.

4. **Scheduled, not ad-hoc.** The `DreamScheduler` enforces regular dream cycles, preventing the agent from running indefinitely without consolidation.

## Metrics

- **Improvement score**: Gap between average success and failure rewards, weighted by replay speed
- **Consolidation compression**: Ratio of experiences to patterns generated
- **Dream frequency**: How often dream cycles trigger relative to normal operation
- **Failure replay coverage**: Percentage of failures that found matching successes
- **ConsolidationReport**: Complete summary of what was learned per cycle

## Testing

The crate includes 12 comprehensive tests covering:
- Dream cycle execution (start, process, complete)
- Failure replay with and without data
- Memory consolidation threshold enforcement
- Pattern generation from mixed experiences
- Scheduler timing and interval accuracy
- Improvement score calculation
- Report generation and formatting
- Cosine similarity correctness
- Multi-tick scheduling with `tick_many`

Run tests with:
```bash
cargo test
```

## Experimental Hypothesis

This crate tests whether periodic offline processing (dreaming) produces better agent improvement than continuous operation. The key prediction: agents that dream about their failures will show faster improvement curves than agents that never pause to consolidate, even though the dreaming agent spends some ticks "offline."

## License

Experimental / research use.
