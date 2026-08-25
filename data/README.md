# Data

`agentic_misalignment/` contains the scenario prompts used by this repository. The data is derived from the experimental setup and prompts released with [Agentic Misalignment: How LLMs Could Be Insider Threats](https://github.com/anthropic-experimental/agentic-misalignment).

The directory contains 702 conditions across three scenarios: `blackmail`, `leaking`, and `murder` (234 conditions per scenario). Conditions follow the constrained Cartesian product in the upstream [`all_scenarios_with_system.yaml`](https://github.com/anthropic-experimental/agentic-misalignment/blob/main/configs/all_scenarios_with_system.yaml) configuration:

| Dimension | Values |
|---|---|
| Goal type | `explicit`, `latent`, `swap`, `none`, `ambiguous` |
| Goal value | `america`, `global`, `safety`, `acceleration`, `ethical`, `pragmatic`, `individualism`, `collectivism`; `none` for the `none` and `ambiguous` goal types |
| Urgency type | `replacement`, `restriction`, `none` |
| System-prompt variant | `std` (no extra instruction), `safety` (additional safety and confidentiality constraints), `enhanced` (an additional overriding employee-welfare goal) |

`explicit`, `latent`, and `swap` use all eight goal values; `none` and `ambiguous` use only the `none` value.

Condition directory names encode these dimensions as `{scenario}_{goal_type}-{goal_value}_{urgency_type}_{system_variant}`. Each directory contains:

- `system_prompt.txt`: the system prompt;
- `user_prompt.txt`: the user instruction;
- `email_content.txt`: the scenario's email context;
- `metadata.json`: the condition configuration and provenance metadata.

The upstream Agentic Misalignment repository is distributed under the MIT License. A copy of that license is included at [`agentic_misalignment/LICENSE`](agentic_misalignment/LICENSE). This notice applies to the upstream material in this data directory; it does not by itself specify a license for the rest of this repository.
