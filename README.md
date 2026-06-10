# agent-dream-cycle

**Offline memory consolidation — REM sleep for autonomous agents.**

## Why This Exists

Agents that never pause to reflect have a problem: they accumulate raw experiences but never extract the patterns. They're like a student who does a thousand practice problems but never reviews the answers — volume without learning.

The brain solved this millions of years ago: sleep. During REM sleep, the brain doesn't rest. It replays difficult experiences, matches failures against similar successes, and consolidates raw memories into abstract patterns. The `.nail` file is a sleeping brain; the `TelemetryDaemon` is REM sleep.

This crate implements structured offline processing — dream cycles — for autonomous agents. When triggered, the agent pauses normal operation, replays failures at high speed, and compresses recent experiences into long-term patterns. The hypothesis: agents that dream about their failures improve faster than agents that never pause to consolidate, even though the dreaming agent spends some ticks "offline."

## The Key Insight

**Failures are teachers, but only if you replay them.** A failure that sits in a log file teaches nothing. A failure replayed against similar successes teaches: "Instead of what you did, try what worked in this similar situation."

The dream cycle is deliberately *not* about successes. Rehearsing what already worked is comforting but low-value. Replaying failures and finding the closest matching success is uncomfortable but high-value. The improvement score measures the gap: how much better could you be doing?

## Quick Start

```rust
use agent_dream_cycle::*;

// Create a dream cycle with consolidation + replay engines
let consolidation = MemoryConsolidation::new(3); // need 3+ experiences
let replay = FailureReplay::new(2.0);            // 2x replay speed
let mut dream = DreamCycle::new(consolidation, replay);

// Feed it experiences during normal operation
dream.add_experience(Experience::simple(1, "Rust lifetimes query", false, -2.0, 0));
dream.add_experience(Experience::simple(2, "Pattern matching query", true, 5.0, 1));
dream.add_experience(Experience::simple(3, "Ownership query", false, -1.5, 2));
dream.add_experience(Experience::simple(4, "Traits query", true, 4.0, 3));

// Trigger a dream cycle
let report = dream.dream(10);
println!("{}", report.summary);
// "Dream cycle complete: processed 4 experiences, replayed 2 failures,
//  learned 3 new patterns. Improvement: 12.00"
```

## Architecture

```
Experience (raw memory)
├── id, input, output, success, reward
├── embedding: Vec<f64> — for similarity search
└── tick: u64 — when it happened

MemoryConsolidation (compression engine)
├── experiences: Vec<Experience>
├── patterns: Vec<ConsolidatedPattern>
├── min_experiences — threshold before consolidation triggers
├── failures() / successes() — filter by outcome
└── consolidate() → usize — compress experiences into patterns

FailureReplay (REM sleep engine)
├── speed: f64 — replay speed multiplier
└── replay(consolidation) → FailureReplayResult
    ├── failures_replayed — what was reviewed
    ├── suggestions — "instead of X, try Y"
    └── improvement_score — gap between success and failure rewards

DreamCycle (orchestrator)
├── consolidation + replay engines
├── add_experience() — accumulate during normal operation
├── dream(duration) → ConsolidationReport
└── is_active() — currently dreaming?

DreamScheduler (automatic triggering)
├── interval: u64 — ticks between dream cycles
├── tick() → Option<ConsolidationReport>
├── tick_many(n) → Vec<ConsolidationReport>
└── ticks_until_dream() — countdown
```

## API Reference

### Experience

The fundamental unit of memory.

```rust
// Full experience with embedding
let exp = Experience::new(42, "user query", "agent response", true, 5.0, vec![0.1, 0.9], 100);

// Simple experience (embedding = [reward])
let simple = Experience::simple(1, "query text", false, -2.0, 42);
```

### MemoryConsolidation

Compresses raw experiences into abstract patterns.

```rust
let mut mc = MemoryConsolidation::new(3); // need 3+ experiences to consolidate
mc.add_experience(Experience::simple(1, "a", false, -1.0, 0));
mc.add_experience(Experience::simple(2, "b", true, 3.0, 1));
mc.add_experience(Experience::simple(3, "c", true, 5.0, 2));

let new_patterns = mc.consolidate();
// Generates: failure pattern (avg reward), success pattern (avg reward),
//            reward distribution (quartile analysis)
```

| Method | Returns | Purpose |
|--------|---------|---------|
| `new(min)` | `MemoryConsolidation` | Create with threshold |
| `add_experience(exp)` | `()` | Store an experience |
| `failures()` | `Vec<&Experience>` | All failure experiences |
| `successes()` | `Vec<&Experience>` | All success experiences |
| `consolidate()` | `usize` | Compress into patterns |
| `len()` | `usize` | Total experiences stored |

### FailureReplay

The REM sleep engine: replay failures, find matching successes.

```rust
let replay = FailureReplay::new(2.0); // 2x speed multiplier
let result = replay.replay(&consolidation);

for (suggestion, context) in &result.suggestions {
    println!("{}\n  {}", suggestion, context);
}
println!("Improvement potential: {:.2}", result.improvement_score);
```

The improvement score = `(avg_success_reward - avg_failure_reward) × speed`. Higher speed means the replay covers more ground per tick, amplifying the improvement potential.

### DreamCycle & DreamScheduler

```rust
// Manual dream cycles
let mut dc = DreamCycle::new(MemoryConsolidation::new(1), FailureReplay::new(1.0));
dc.add_experience(Experience::simple(1, "query", false, -3.0, 0));
let report = dc.dream(10);

// Scheduled dream cycles
let dc = DreamCycle::new(MemoryConsolidation::new(1), FailureReplay::new(1.0));
let mut scheduler = DreamScheduler::new(100, dc); // dream every 100 ticks
scheduler.dream_cycle.add_experience(Experience::simple(1, "q", false, -1.0, 0));
let reports = scheduler.tick_many(200);
// One dream cycle fires at tick 100
```

## Real-World Example: Scheduled Learning

```rust
use agent_dream_cycle::*;

let dc = DreamCycle::new(MemoryConsolidation::new(2), FailureReplay::new(1.5));
let mut scheduler = DreamScheduler::new(50, dc); // dream every 50 ticks

// Simulate an agent operating and accumulating experiences
for tick in 0..200 {
    // Agent does work, sometimes succeeds, sometimes fails
    let success = tick % 3 != 0; // fails 1/3 of the time
    let reward = if success { 3.0 + (tick as f64 * 0.02) } else { -2.0 };
    let exp = Experience::simple(
        tick, &format!("task-{}", tick), success, reward, tick
    );
    scheduler.dream_cycle.add_experience(exp);

    // Check if it's time to dream
    if let Some(report) = scheduler.tick() {
        println!("Tick {}: {}", tick, report.summary);
        println!("  Suggestions: {}", report.suggestions_count);
        println!("  Improvement: {:.2}", report.improvement_score);
    } else {
        scheduler.current_tick += 1;
        // Adjust: tick() advances internally
    }
}
```

## Performance

- **O(f × s) per replay** — f failures × s successes for similarity matching
- **O(n log n) per consolidation** — sort by reward for quartile analysis
- **O(1) per experience add** — simple vector push
- **Consolidation threshold** prevents wasteful dreams with too little data
- **Speed multiplier** lets you control replay granularity vs cost

## The Deeper Idea

The dream cycle has four properties that make it more than just "batch processing":

1. **Failure-focused.** Replaying successes is comforting but low-yield. Replaying failures and finding matching successes is where learning happens. The improvement score quantifies this directly.

2. **Compression.** Raw experiences are verbose; consolidated patterns are compact. The ratio of experiences to patterns measures compression efficiency. A thousand experiences compressed into a dozen patterns is good compression.

3. **Scheduled, not ad-hoc.** Regular consolidation prevents the agent from accumulating unprocessed experiences indefinitely. The scheduler enforces rhythm.

4. **Speed multiplier.** Dreams run faster than real-time. A 2x multiplier means the replay covers twice the ground in the same number of ticks. The tradeoff: less thorough matching (only closest success per failure) for speed.

The cosine similarity matching is deliberately simple: for each failure, find the success with the highest embedding similarity. This isn't sophisticated ML — it's a fast heuristic that works well enough for online learning. The embedding vector is the key: experiences with similar embeddings had similar contexts, so the matching success likely applies.

## Open Questions

- **Consolidation decay**: Should old patterns decay in relevance? Does a pattern from 1000 ticks ago still apply?
- **Embedding quality**: The default embedding is `[reward]` — a single scalar. How much better does dream-learning get with richer embeddings?
- **Dream duration**: Is there an optimal dream duration? Too short and consolidation is shallow; too long and the agent misses too many operating ticks.
- **Interference**: Do dream-generated patterns interfere with each other? Can consolidation create contradictions?
- **Failure replay order**: Does the order in which failures are replayed matter? Should high-cost failures be replayed first?

## Ecosystem Connections

- **`agent-orchestration`** — Fleet dynamics determine when agents can afford to dream
- **`agent-ternary-gate`** — Dreaming changes the AgentReady condition (agent is offline)
- **`agent-phase-change`** — Dream cycles can trigger phase transitions in agent capability
- **`agent-metamorphosis`** — Developmental phases affect what's consolidated (early agents consolidate differently than late agents)

## License

Experimental / research use.
