# I Made Claude Opus 4.6 Sit the JEE Advanced Mathematics Paper. Here's What Happened.

**Subtitle:** Yes, I know what you're thinking — who tests a next-token prediction engine on mathematics? I did. Purely out of curiosity. No regrets.

---

If you grew up in India, you know what JEE Advanced is. If you didn't — it's arguably the hardest undergraduate entrance exam in the world. Roughly 200,000 students qualify just to *sit* it, after a year or more of relentless preparation. The mathematics section is designed to make even brilliant students second-guess themselves.

I gave it to Opus 4.6 blind — no answer key, no hints, just the raw questions. Eight numerical problems from JEE Advanced 2025 Paper 2.

It got all eight correct.

---

## The Setup

- **Model:** claude-opus-4-6 with extended thinking, effort set to maximum
- **Input:** LaTeX question text only — no worked examples, no answer choices
- **Scoring:** JEE-style, answers verified against the official key
- **Total cost:** $0.79

Eight questions. Eight correct answers. Under a dollar.

---

## The Scoreboard

| Question | Topic | Time | Correct |
|----------|-------|------|---------|
| Q9 | Differential equations | 25.7s | ✓ |
| Q10 | Binomial coefficients | 42.9s | ✓ |
| Q11 | Probability | 27.1s | ✓ |
| Q12 | 3D Vectors | 47.5s | ✓ |
| Q13 | Complex numbers | 40.3s | ✓ |
| Q14 | Composite functions | 21.0s | ✓ |
| Q15 | Trigonometric sums | **92.8s** | ✓ |
| Q16 | Definite integrals | 34.1s | ✓ |

---

## The One That Made It Sweat

Q15 took **92 seconds** — nearly 4× longer than the fastest question. It involved a telescoping trigonometric sum where the trick is recognising a sine difference identity buried inside products of consecutive sines. Students who've seen this pattern before get it in two minutes. Those who haven't spend twenty.

Opus spent 92 seconds and still got there. The [full working is here](equations.html#q15).

---

## The One That Revealed Something Interesting

Q12 — a vectors coplanarity problem — was where things got genuinely curious. Inside the thinking trace, the model arrived at a solution, then silently revisited its own setup, caught an inconsistency in how it had framed the coplanarity condition, corrected it, and arrived at the right answer: **−2**.

No prompt told it to check its work. It just did.

Call that "next-token prediction" if you want. The label stops being satisfying when the token sequence looks like this.

The [equations and working for Q12 are here](equations.html#q12).

---

## What This Means (Or Doesn't)

Opus 4.6 never sat in a coaching class in Kota. It has no stake in the result, no 3-hour clock ticking, no anxiety. It spent less than $1 on a paper that students spend years preparing for.

That's either deeply impressive or deeply unsettling, depending on who you ask.

The honest answer is probably both.

Is it *just* a next-token predictor? Technically, in the same way you're *just* neurons firing. At some level of abstraction the label is true. At the level that matters — does it solve the problem, does it catch its own errors, does it produce verifiable reasoning — the label starts to feel like it's doing more concealing than explaining.

I'm not here to tell you what to conclude. The scoreboard is 8/8.

---

*All raw API responses, thinking traces, and timings are in the [GitHub repo](https://github.com/adithyagiridharan/Anthropic). The rendered equations for all eight questions are [here](equations.html).*
