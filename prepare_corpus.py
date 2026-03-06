"""Build tokenizer training corpus from existing data + synthetic STEM text."""

import json
import os


def load_jsonl_texts(data_dir: str = "data") -> list[str]:
    """Load all text from data/*.jsonl files."""
    texts = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".jsonl"):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    texts.append(obj["text"])
    return texts


def generate_synthetic_stem() -> list[str]:
    """Generate synthetic STEM text for vocabulary coverage."""
    lines = []

    # Chemical formulas
    formulas = [
        "H2O water molecule consists of two hydrogen atoms and one oxygen atom",
        "NaCl sodium chloride is an ionic compound formed by Na+ and Cl- ions",
        "CH3COOH acetic acid is a weak organic acid with pKa of 4.76",
        "C6H12O6 glucose is a monosaccharide with the molecular formula C6H12O6",
        "H2SO4 sulfuric acid is a strong diprotic acid",
        "CO2 carbon dioxide is a linear molecule with two C=O double bonds",
        "NH3 ammonia has a trigonal pyramidal molecular geometry",
        "CaCO3 calcium carbonate decomposes into CaO and CO2 when heated",
        "Fe2O3 iron(III) oxide is commonly known as rust",
        "CH4 methane is the simplest alkane hydrocarbon",
        "C2H5OH ethanol is a primary alcohol with hydroxyl group",
        "HCl hydrochloric acid fully dissociates in aqueous solution",
        "NaOH sodium hydroxide is a strong base",
        "KMnO4 potassium permanganate is a strong oxidizing agent",
        "ATP adenosine triphosphate stores cellular energy",
        "DNA deoxyribonucleic acid carries genetic information",
        "RNA ribonucleic acid translates genetic code into proteins",
    ]
    lines.extend(formulas)

    # Physics equations and notation
    physics = [
        "F=ma force equals mass times acceleration Newton's second law",
        "E=mc^2 energy equals mass times the speed of light squared",
        "PV=nRT the ideal gas law relates pressure volume and temperature",
        "F=kx Hooke's law states that force is proportional to displacement",
        "v=u+at velocity equals initial velocity plus acceleration times time",
        "s=ut+0.5at^2 displacement with constant acceleration",
        "KE=0.5mv^2 kinetic energy equals half mass times velocity squared",
        "PE=mgh gravitational potential energy equals mass times g times height",
        "W=Fd work equals force times displacement",
        "P=W/t power equals work divided by time",
        "F=Gm1m2/r^2 Newton's law of universal gravitation",
        "E=hf energy of a photon equals Planck's constant times frequency",
        "lambda=h/p de Broglie wavelength relates wavelength to momentum",
        "dv/dt represents the rate of change of velocity with respect to time",
        "dp/dt=F the time derivative of momentum equals force",
        "nabla x E = -dB/dt Faraday's law of electromagnetic induction",
        "nabla . B = 0 Gauss's law for magnetism no magnetic monopoles",
        "S=kB*ln(W) Boltzmann entropy formula",
        "F=-dU/dr force is the negative gradient of potential energy",
        "tau=r x F torque is the cross product of position and force",
        "L=r x p angular momentum is the cross product of position and momentum",
        "I=integral(r^2 dm) moment of inertia",
        "Schrodinger equation: i*hbar*d/dt|psi>=H|psi>",
        "Heisenberg uncertainty: delta_x * delta_p >= hbar/2",
        "Maxwell-Boltzmann distribution: f(v) = 4*pi*(m/2*pi*kT)^(3/2)*v^2*exp(-mv^2/2kT)",
    ]
    lines.extend(physics)

    # Math expressions and concepts
    math_text = [
        "the integral of f(x)dx from a to b equals F(b) minus F(a)",
        "the derivative of x^n is n*x^(n-1) by the power rule",
        "eigenvalue lambda satisfies Av = lambda*v for eigenvector v",
        "the determinant of a 2x2 matrix [[a,b],[c,d]] is ad-bc",
        "matrix multiplication is associative but not commutative",
        "the trace of a matrix equals the sum of its eigenvalues",
        "gradient descent updates w = w - alpha * dL/dw",
        "the chain rule states d/dx[f(g(x))] = f'(g(x)) * g'(x)",
        "Taylor series: f(x) = sum_{n=0}^{inf} f^(n)(a)/n! * (x-a)^n",
        "Fourier transform: F(w) = integral f(t)*e^(-iwt) dt",
        "Euler's identity: e^(i*pi) + 1 = 0",
        "the quadratic formula: x = (-b +/- sqrt(b^2 - 4ac)) / (2a)",
        "Bayes theorem: P(A|B) = P(B|A)*P(A)/P(B)",
        "the binomial coefficient C(n,k) = n! / (k! * (n-k)!)",
        "the Cauchy-Schwarz inequality: |<u,v>|^2 <= <u,u>*<v,v>",
        "a vector space requires closure under addition and scalar multiplication",
        "the rank-nullity theorem: rank(A) + nullity(A) = n",
        "convergence of a series requires lim_{n->inf} a_n = 0",
        "the Jacobian matrix contains all first-order partial derivatives",
        "Lagrange multipliers: nabla f = lambda * nabla g at constrained extrema",
        "the Laplacian operator: nabla^2 = d^2/dx^2 + d^2/dy^2 + d^2/dz^2",
        "divergence theorem: integral_V (nabla . F) dV = integral_S F . dA",
        "Stokes theorem: integral_S (nabla x F) . dA = integral_C F . dr",
    ]
    lines.extend(math_text)

    # Scientific units
    units = [
        "SI base units: kilogram kg, meter m, second s, ampere A, kelvin K, mole mol, candela cd",
        "derived units: newton N = kg*m/s^2, joule J = N*m, watt W = J/s",
        "pressure: pascal Pa = N/m^2, atmosphere atm = 101325 Pa",
        "electric: volt V = J/C, ohm Omega = V/A, farad F = C/V",
        "magnetic: tesla T = kg/(A*s^2), henry H = V*s/A",
        "concentration: mol/L = M (molar), parts per million ppm",
        "energy: electronvolt eV = 1.602e-19 J, calorie cal = 4.184 J",
        "frequency: hertz Hz = 1/s, angular frequency omega = 2*pi*f rad/s",
        "speed of light c = 299792458 m/s = 3.00e8 m/s",
        "Planck constant h = 6.626e-34 J*s, hbar = h/(2*pi)",
        "Boltzmann constant kB = 1.381e-23 J/K",
        "Avogadro number NA = 6.022e23 mol^-1",
        "gravitational constant G = 6.674e-11 N*m^2/kg^2",
        "elementary charge e = 1.602e-19 C",
        "permittivity of free space epsilon_0 = 8.854e-12 F/m",
        "permeability of free space mu_0 = 4*pi*1e-7 H/m",
    ]
    lines.extend(units)

    # Greek letters in context
    greek = [
        "alpha decay: nucleus emits an alpha particle (helium-4 nucleus)",
        "beta coefficient measures systematic risk in finance",
        "gamma rays have the shortest wavelength in the electromagnetic spectrum",
        "delta represents a small change: delta_x approaches zero in limits",
        "epsilon-delta definition of limits: for all epsilon>0 exists delta>0",
        "zeta function: zeta(s) = sum_{n=1}^{inf} 1/n^s for Re(s)>1",
        "eta efficiency = useful output / total input",
        "theta is the standard variable for angles in polar coordinates",
        "iota is used for the imaginary unit in some notations",
        "kappa curvature measures how fast a curve changes direction",
        "lambda wavelength in optics, eigenvalue in linear algebra",
        "mu is used for the coefficient of friction and micro prefix (1e-6)",
        "nu frequency in hertz, kinematic viscosity in fluid mechanics",
        "xi is a common variable for dimensionless coordinates",
        "pi = 3.14159265358979... the ratio of circumference to diameter",
        "rho density = mass/volume, charge density in electrostatics",
        "sigma standard deviation, stress in mechanics, Stefan-Boltzmann constant",
        "tau torque, time constant in RC circuits tau = RC",
        "phi electric potential, golden ratio phi = (1+sqrt(5))/2 = 1.618...",
        "chi-squared test statistic for goodness of fit",
        "psi wave function in quantum mechanics",
        "omega angular velocity = 2*pi*f rad/s",
    ]
    lines.extend(greek)

    # Numbers in varied formats
    numbers = [
        "pi = 3.14159265358979323846...",
        "e = 2.71828182845904523536...",
        "sqrt(2) = 1.41421356237...",
        "the fine structure constant alpha = 1/137.036 = 7.297e-3",
        "Avogadro's number NA = 6.02214076e23 per mole",
        "Planck length = 1.616255e-35 meters",
        "age of the universe = 13.8 billion years = 4.35e17 seconds",
        "mass of electron = 9.109e-31 kg = 0.511 MeV/c^2",
        "mass of proton = 1.673e-27 kg = 938.3 MeV/c^2",
        "Bohr radius a0 = 5.292e-11 m = 0.529 angstroms",
        "room temperature = 293.15 K = 20 C = 68 F",
        "standard pressure = 101325 Pa = 1 atm = 760 mmHg",
        "water density = 997 kg/m^3 at 25 C",
        "speed of sound in air = 343 m/s at 20 C",
        "1 light-year = 9.461e15 meters",
        "1 astronomical unit AU = 1.496e11 meters",
        "absolute zero = 0 K = -273.15 C",
        "12345 67890 individual digits should each be separate tokens",
        "0 1 2 3 4 5 6 7 8 9 are the ten decimal digits",
        "binary: 1010 = 10, hexadecimal: 0xFF = 255, octal: 0o77 = 63",
    ]
    lines.extend(numbers)

    # LaTeX-style notation
    latex = [
        "\\frac{dy}{dx} represents the derivative of y with respect to x",
        "\\sum_{i=1}^{n} a_i is the sum of a_i from i=1 to n",
        "\\int_{a}^{b} f(x) dx is the definite integral from a to b",
        "\\prod_{i=1}^{n} a_i is the product of a_i from i=1 to n",
        "\\lim_{x \\to 0} \\frac{\\sin x}{x} = 1",
        "\\nabla f = (\\partial f/\\partial x, \\partial f/\\partial y, \\partial f/\\partial z)",
        "\\vec{F} = q(\\vec{E} + \\vec{v} \\times \\vec{B}) is the Lorentz force",
        "\\binom{n}{k} = \\frac{n!}{k!(n-k)!}",
        "\\sqrt{a^2 + b^2} is the hypotenuse length by Pythagorean theorem",
        "\\hat{H}\\psi = E\\psi is the time-independent Schrodinger equation",
        "\\mathbb{R}^n represents n-dimensional real vector space",
        "\\forall x \\in \\mathbb{R}, \\exists n \\in \\mathbb{N} : n > x (Archimedean property)",
    ]
    lines.extend(latex)

    # Biology terms
    biology = [
        "photosynthesis converts CO2 and H2O into glucose C6H12O6 and O2 using light energy",
        "cellular respiration: C6H12O6 + 6O2 -> 6CO2 + 6H2O + 38 ATP",
        "DNA replication is semiconservative: each strand serves as template",
        "transcription: DNA -> mRNA in the nucleus by RNA polymerase",
        "translation: mRNA -> protein at ribosomes using tRNA anticodons",
        "mitosis produces two identical diploid daughter cells",
        "meiosis produces four haploid gametes with genetic recombination",
        "natural selection acts on phenotypic variation in populations",
        "Hardy-Weinberg equilibrium: p^2 + 2pq + q^2 = 1 for allele frequencies",
        "enzyme kinetics: v = Vmax*[S]/(Km + [S]) Michaelis-Menten equation",
        "the Krebs cycle (citric acid cycle) generates NADH, FADH2, and GTP",
        "osmosis is the diffusion of water across a semipermeable membrane",
        "electrochemical gradient drives ATP synthesis in chemiosmosis",
        "action potential propagation: resting (-70mV) -> depolarization -> repolarization",
    ]
    lines.extend(biology)

    # Chemistry concepts
    chemistry = [
        "electronegativity increases across a period and decreases down a group",
        "pH = -log10[H+] measures hydrogen ion concentration",
        "oxidation is loss of electrons, reduction is gain of electrons (OIL RIG)",
        "equilibrium constant Keq = [products]/[reactants] at equilibrium",
        "Gibbs free energy: delta_G = delta_H - T*delta_S determines spontaneity",
        "Hess's law: total enthalpy change is independent of pathway",
        "Le Chatelier's principle: system shifts to counteract applied stress",
        "electron configuration of Fe: [Ar] 3d6 4s2, Fe2+: [Ar] 3d6",
        "hybridization: sp3 (tetrahedral 109.5), sp2 (trigonal 120), sp (linear 180)",
        "rate law: rate = k[A]^m[B]^n where m,n are reaction orders",
        "Nernst equation: E = E0 - (RT/nF)*ln(Q) relates potential to concentrations",
        "colligative properties: boiling point elevation delta_Tb = Kb*m*i",
        "bond energy: C-H 413 kJ/mol, C=C 614 kJ/mol, C-C 348 kJ/mol",
    ]
    lines.extend(chemistry)

    # Extended derivations for better context
    derivations = [
        "Deriving kinetic energy from work-energy theorem: "
        "W = integral F dx = integral ma dx = integral m(dv/dt)dx = integral mv dv = 0.5mv^2 - 0.5mv0^2. "
        "Therefore KE = 0.5mv^2.",

        "Deriving the ideal gas law from kinetic theory: "
        "pressure P = (1/3)*rho*v_rms^2 where rho = Nm/V. "
        "Average kinetic energy = (3/2)kT per molecule. "
        "So PV = NkT = nRT where R = NA*k.",

        "The fundamental theorem of calculus connects differentiation and integration: "
        "if F(x) = integral_a^x f(t)dt then F'(x) = f(x). "
        "This means integration is the inverse of differentiation.",

        "Conservation of energy in a closed system: "
        "dE/dt = 0 implies E_initial = E_final. "
        "For a falling object: mgh_1 + 0.5mv_1^2 = mgh_2 + 0.5mv_2^2. "
        "Energy transforms between kinetic and potential forms but total is conserved.",

        "Deriving the wave equation: "
        "consider a string with tension T and linear density mu. "
        "Newton's second law on a small element gives: "
        "T * d^2y/dx^2 = mu * d^2y/dt^2, so v = sqrt(T/mu) is the wave speed.",

        "Entropy and the second law of thermodynamics: "
        "for any spontaneous process, delta_S_universe >= 0. "
        "Boltzmann's statistical interpretation: S = kB * ln(W) "
        "where W is the number of microstates. "
        "A system evolves toward the macrostate with the most microstates.",

        "Deriving the Schwarzschild radius: "
        "set escape velocity equal to speed of light: 0.5mv^2 = GMm/r. "
        "With v=c: r_s = 2GM/c^2. "
        "For Earth: r_s = 2*6.674e-11*5.972e24/(3e8)^2 = 8.87 mm.",

        "Dimensional analysis of the period of a pendulum: "
        "T depends on length L, mass m, and gravity g. "
        "[T] = [L]^a [m]^b [g]^c -> s = m^a * kg^b * (m/s^2)^c. "
        "Solving: a=1/2, b=0, c=-1/2, so T = k*sqrt(L/g). "
        "The mass does not affect the period.",
    ]
    lines.extend(derivations)

    return lines


def main():
    print("Building tokenizer training corpus...")

    # Load existing data
    texts = load_jsonl_texts()
    print(f"  Loaded {len(texts)} examples from data/*.jsonl")

    # Generate synthetic STEM text
    synthetic = generate_synthetic_stem()
    print(f"  Generated {len(synthetic)} synthetic STEM lines")

    # Combine
    all_lines = texts + synthetic

    # Write corpus
    os.makedirs("tokenizer", exist_ok=True)
    corpus_path = os.path.join("tokenizer", "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in all_lines:
            f.write(line + "\n")

    total_chars = sum(len(line) for line in all_lines)
    print(f"  Total: {len(all_lines)} lines, {total_chars:,} characters")
    print(f"  Written to {corpus_path}")


if __name__ == "__main__":
    main()
