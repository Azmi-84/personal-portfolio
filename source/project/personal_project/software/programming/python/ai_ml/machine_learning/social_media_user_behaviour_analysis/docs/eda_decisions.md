# EDA Decisions Log

> Move the content of your existing `resources/docs/eda_decisions.md` in
> here (or just copy your existing file over this template - the point is
> to keep one canonical decisions log the config file can point back to).

For every feature, record:

| Feature | Kept? | Reason |
|---|---|---|
| age | Yes | Strong correlation with target in profiling report |
| user_id | No | Identifier, no predictive signal |
| ... | ... | ... |

This file is the paper trail `configs/data_config.yaml`'s
`selected_features` / `dropped_features` should always match.
