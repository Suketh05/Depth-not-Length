// TheoryTests.cs -- Cross-language + cross-check-the-paper suite for the depth theory.
//
// Two kinds of assertion, both from authoritative sources (never from running this code):
//   (1) Section-A THEORY goldens -- full-precision float64 values from the canonical golden
//       pack, shared verbatim with the Python reference (src/membench/theory/) and the Java
//       arm, so a disagreement means a bug in one derivation.
//   (2) Section-B PAPER-TABLE identities -- numbers a reader can re-derive from the paper's
//       own published tables (tab:realret use factors, the Fano 1/K floor, tab:scorecard).
//
// Tolerance policy (golden pack): exact-fraction goldens are asserted essentially exactly;
// float64 goldens use abs err <= 1e-12 (relaxed to 1e-9 for a ratio of small numbers). The
// published-kappa identity uses xunit's decimal-place overload to assert the 3-dp rounding.
using Xunit;

namespace Membench;

public class TheoryTests
{
    private static readonly SurvivalModel Calib = Theory.Calibration; // s0=0.70, rho=0.67, q=0.97
    private const double Tol = 1e-12;
}
