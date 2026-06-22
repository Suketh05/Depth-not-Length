<!-- Brief vs Mem0 COMPLETE head-to-head (synthetic, GPT answer model) | head-to-head | measured -->

**Table 103. Brief vs Mem0 head-to-head (synthetic, GPT-5.1, compliance by depth)**

| arm | d1 | d2 | d3 | all |
|---|---|---|---|---|
| Brief | 1.00 | 1.00 | 1.00 | 1.00 |
| Mem0 | 1.00 | 0.93 | 0.93 | 0.96 |
| dense | 0.95 | 0.90 | 0.60 | 0.82 |
| none | 0.00 | 0.00 | 0.00 | 0.00 |

_Mem0 ran reliably (45/45, sequential). Brief holds 1.00 flat across depth; Mem0 is strong (0.96) but decays slightly at depth (1.00→0.93). Both far above dense/none. Brief's edge is small, real, and at depth — consistent with link-following being perfectly depth-stable vs Mem0's LLM-extraction decay._
