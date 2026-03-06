"""Build tokenizer training corpus from real STEM datasets + synthetic text.

Downloads ~5-10MB from HuggingFace (streaming, no full dataset download):
- OpenWebMath (web math content)
- Proof-Pile-2 arXiv subset (math/CS papers)
- Big-Math (verified competition problems)

Requires: pip install datasets
"""

import json
import os


# --- Real data from HuggingFace ---

def download_openwebmath(target_chars: int = 5_000_000) -> list[str]:
    """Stream OpenWebMath examples."""
    from datasets import load_dataset
    texts = []
    total = 0
    print("    Streaming open-web-math/open-web-math ...")
    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    for example in ds:
        text = example.get("text", "")
        if len(text) < 50 or len(text) > 10000:
            continue
        texts.append(text)
        total += len(text)
        if total >= target_chars:
            break
    print(f"    Got {len(texts)} examples, {total:,} chars")
    return texts


def download_proofpile2(target_chars: int = 5_000_000) -> list[str]:
    """Stream Proof-Pile-2 arXiv subset."""
    from datasets import load_dataset
    texts = []
    total = 0
    print("    Streaming EleutherAI/proof-pile-2 (arxiv) ...")
    try:
        ds = load_dataset(
            "EleutherAI/proof-pile-2",
            name="arxiv",
            split="train",
            streaming=True,
        )
    except Exception:
        # Fallback: try without name/config
        try:
            ds = load_dataset(
                "EleutherAI/proof-pile-2",
                split="train",
                streaming=True,
            )
        except Exception as e:
            print(f"    SKIP: Could not load proof-pile-2: {e}")
            return []
    for example in ds:
        text = example.get("text", "")
        if len(text) < 100 or len(text) > 10000:
            continue
        texts.append(text)
        total += len(text)
        if total >= target_chars:
            break
    print(f"    Got {len(texts)} examples, {total:,} chars")
    return texts


def download_bigmath(target_chars: int = 5_000_000) -> list[str]:
    """Stream Big-Math verified competition problems."""
    from datasets import load_dataset
    texts = []
    total = 0
    print("    Streaming SynthLabsAI/Big-Math ...")
    try:
        ds = load_dataset("SynthLabsAI/Big-Math", split="train", streaming=True)
    except Exception as e:
        print(f"    SKIP: Could not load Big-Math: {e}")
        return []
    for example in ds:
        # Big-Math may have 'problem', 'solution', or 'text' fields
        text = example.get("text", "")
        if not text:
            parts = []
            if example.get("problem"):
                parts.append(example["problem"])
            if example.get("solution"):
                parts.append(example["solution"])
            text = "\n".join(parts)
        if len(text) < 30 or len(text) > 10000:
            continue
        texts.append(text)
        total += len(text)
        if total >= target_chars:
            break
    print(f"    Got {len(texts)} examples, {total:,} chars")
    return texts


def download_algebraicstack(target_chars: int = 3_000_000) -> list[str]:
    """Stream AlgebraicStack (verified math code)."""
    from datasets import load_dataset
    texts = []
    total = 0
    print("    Streaming EleutherAI/proof-pile-2 (algebraic-stack) ...")
    try:
        ds = load_dataset(
            "EleutherAI/proof-pile-2",
            name="algebraic-stack",
            split="train",
            streaming=True,
        )
    except Exception as e:
        print(f"    SKIP: Could not load algebraic-stack: {e}")
        return []
    for example in ds:
        text = example.get("text", "")
        if len(text) < 100 or len(text) > 8000:
            continue
        texts.append(text)
        total += len(text)
        if total >= target_chars:
            break
    print(f"    Got {len(texts)} examples, {total:,} chars")
    return texts


# --- Local data ---

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


# --- Synthetic STEM text (notation coverage) ---

def generate_synthetic_stem() -> list[str]:
    """Generate synthetic STEM text for notation/formula coverage."""
    lines = []

    # Chemical formulas
    lines.extend([
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
        "SiO2 silicon dioxide is the primary component of quartz and glass",
        "Al2O3 aluminum oxide or alumina has a very high melting point of 2072 C",
        "MgCl2 magnesium chloride dissociates into Mg2+ and 2 Cl- ions",
        "K2Cr2O7 potassium dichromate is a powerful oxidizing agent in acidic solutions",
        "Na2CO3 sodium carbonate is commonly called washing soda or soda ash",
        "Ca(OH)2 calcium hydroxide is also known as slaked lime",
        "NH4NO3 ammonium nitrate is used as fertilizer and in explosives",
        "Cu2+ copper(II) ion has an electron configuration of [Ar] 3d9",
        "C2H4 ethylene is the simplest alkene with one C=C double bond",
        "C3H8 propane is a three-carbon alkane used as fuel gas",
        "HCOOH formic acid is the simplest carboxylic acid",
        "C6H5OH phenol is an aromatic compound with a hydroxyl group on benzene",
        "PCl5 phosphorus pentachloride has a trigonal bipyramidal geometry",
    ])

    # Physics equations and notation
    lines.extend([
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
        "Coulomb's law: F = k*q1*q2/r^2 where k = 8.99e9 N*m^2/C^2",
        "Gauss's law: integral E . dA = Q_enclosed / epsilon_0",
        "Ampere's law: integral B . dl = mu_0 * I_enclosed",
        "Snell's law: n1*sin(theta1) = n2*sin(theta2) relates refraction angles",
        "Stefan-Boltzmann law: P = sigma*A*T^4 for blackbody radiation",
        "Wien's displacement law: lambda_max = b/T where b = 2.898e-3 m*K",
        "Compton scattering: delta_lambda = (h/m_e*c)*(1-cos(theta))",
        "Bragg's law: 2*d*sin(theta) = n*lambda for X-ray diffraction",
        "Lorentz force: F = q(E + v x B) for charged particle in EM field",
        "Relativistic energy: E^2 = (pc)^2 + (mc^2)^2",
        "Time dilation: t' = t / sqrt(1 - v^2/c^2) = gamma * t",
        "Length contraction: L' = L * sqrt(1 - v^2/c^2) = L / gamma",
        "Doppler effect: f' = f * (v +/- v_observer) / (v -/+ v_source)",
        "Bernoulli's equation: P + 0.5*rho*v^2 + rho*g*h = constant along streamline",
        "Reynolds number: Re = rho*v*L/mu determines laminar vs turbulent flow",
        "Navier-Stokes: rho*(dv/dt + v.nabla v) = -nabla P + mu*nabla^2 v + f",
        "Poisson's equation: nabla^2 phi = -rho/epsilon_0 for electrostatic potential",
        "Laplace's equation: nabla^2 phi = 0 in charge-free regions",
        "Kirchhoff's voltage law: sum of voltages around any closed loop equals zero",
        "Kirchhoff's current law: sum of currents at any junction equals zero",
    ])

    # Math expressions and concepts
    lines.extend([
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
        "Green's theorem: integral_C (P dx + Q dy) = integral_D (dQ/dx - dP/dy) dA",
        "the Gram-Schmidt process orthogonalizes a set of linearly independent vectors",
        "singular value decomposition: A = U*Sigma*V^T where U,V orthogonal",
        "the spectral theorem: every real symmetric matrix is orthogonally diagonalizable",
        "Jordan normal form decomposes a matrix into Jordan blocks along the diagonal",
        "Cayley-Hamilton theorem: every square matrix satisfies its own characteristic equation",
        "the fundamental group pi_1(S^1) = Z classifies loops on the circle up to homotopy",
        "Riemann integral: lim_{n->inf} sum_{i=1}^n f(x_i*) * delta_x_i",
        "Lebesgue integral generalizes Riemann by measuring sets rather than intervals",
        "contour integration: integral_C f(z) dz = 2*pi*i * sum of residues inside C",
        "the residue theorem connects complex integration to local behavior at poles",
        "Galois theory connects field extensions to group theory via automorphisms",
        "the Chinese remainder theorem: system of congruences with coprime moduli has unique solution",
        "Fermat's little theorem: a^(p-1) = 1 mod p for prime p and gcd(a,p)=1",
        "the fundamental theorem of arithmetic: every integer > 1 has unique prime factorization",
        "Zorn's lemma: if every chain in a partially ordered set has an upper bound then the set has a maximal element",
        "the axiom of choice is equivalent to Zorn's lemma and the well-ordering principle",
    ])

    # Scientific units and constants
    lines.extend([
        "SI base units: kilogram kg, meter m, second s, ampere A, kelvin K, mole mol, candela cd",
        "derived units: newton N = kg*m/s^2, joule J = N*m, watt W = J/s",
        "pressure: pascal Pa = N/m^2, atmosphere atm = 101325 Pa",
        "electric: volt V = J/C, ohm Omega = V/A, farad F = C/V",
        "magnetic: tesla T = kg/(A*s^2), henry H = V*s/A",
        "concentration: mol/L = M (molar), parts per million ppm",
        "energy: electronvolt eV = 1.602e-19 J, calorie cal = 4.184 J",
        "frequency: hertz Hz = 1/s, angular frequency omega = 2*pi*f rad/s",
        "speed of light c = 299792458 m/s = 3.00e8 m/s",
        "Planck constant h = 6.626e-34 J*s, hbar = h/(2*pi) = 1.055e-34 J*s",
        "Boltzmann constant kB = 1.381e-23 J/K",
        "Avogadro number NA = 6.022e23 mol^-1",
        "gravitational constant G = 6.674e-11 N*m^2/kg^2",
        "elementary charge e = 1.602e-19 C",
        "permittivity of free space epsilon_0 = 8.854e-12 F/m",
        "permeability of free space mu_0 = 4*pi*1e-7 H/m = 1.257e-6 H/m",
        "Rydberg constant R_inf = 1.097e7 m^-1",
        "gas constant R = 8.314 J/(mol*K) = NA * kB",
        "Faraday constant F = 96485 C/mol = NA * e",
        "Stefan-Boltzmann constant sigma = 5.670e-8 W/(m^2*K^4)",
        "fine structure constant alpha = e^2/(4*pi*epsilon_0*hbar*c) = 1/137.036",
        "Bohr magneton mu_B = e*hbar/(2*m_e) = 9.274e-24 J/T",
    ])

    # Greek letters in context
    lines.extend([
        "alpha decay: nucleus emits an alpha particle (helium-4 nucleus)",
        "beta coefficient measures systematic risk in finance and portfolio theory",
        "gamma rays have the shortest wavelength in the electromagnetic spectrum",
        "delta represents a small change: delta_x approaches zero in limits",
        "epsilon-delta definition of limits: for all epsilon>0 exists delta>0",
        "zeta function: zeta(s) = sum_{n=1}^{inf} 1/n^s for Re(s)>1",
        "eta efficiency = useful output / total input expressed as percentage",
        "theta is the standard variable for angles in polar coordinates",
        "kappa curvature measures how fast a curve changes direction",
        "lambda wavelength in optics, eigenvalue in linear algebra",
        "mu is used for the coefficient of friction and micro prefix (1e-6)",
        "nu frequency in hertz, kinematic viscosity in fluid mechanics",
        "xi is a common variable for dimensionless coordinates in PDEs",
        "pi = 3.14159265358979... the ratio of circumference to diameter",
        "rho density = mass/volume in kg/m^3, charge density in electrostatics",
        "sigma standard deviation, stress in mechanics Pa, Stefan-Boltzmann constant",
        "tau torque in N*m, time constant in RC circuits tau = RC",
        "phi electric potential in volts, golden ratio phi = (1+sqrt(5))/2 = 1.618...",
        "chi-squared test statistic for goodness of fit in statistics",
        "psi wave function in quantum mechanics, psi(x,t)",
        "omega angular velocity = 2*pi*f rad/s, solid angle in steradians",
    ])

    # Numbers in varied formats
    lines.extend([
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
        "1 light-year = 9.461e15 meters = 63241 AU",
        "1 astronomical unit AU = 1.496e11 meters",
        "absolute zero = 0 K = -273.15 C = -459.67 F",
        "12345 67890 individual digits should each be separate tokens",
        "0 1 2 3 4 5 6 7 8 9 are the ten decimal digits",
        "binary: 1010 = 10, hexadecimal: 0xFF = 255, octal: 0o77 = 63",
    ])

    # LaTeX-style notation
    lines.extend([
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
        "\\iint_D f(x,y) dA = \\int_a^b \\int_{g(x)}^{h(x)} f(x,y) dy dx",
        "\\oint_C \\vec{F} \\cdot d\\vec{r} represents the circulation of F around C",
        "\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix} is a 2x2 matrix",
        "\\frac{\\partial^2 u}{\\partial t^2} = c^2 \\frac{\\partial^2 u}{\\partial x^2} wave equation",
        "\\sum_{n=0}^{\\infty} \\frac{x^n}{n!} = e^x the exponential series",
        "\\det(A - \\lambda I) = 0 the characteristic equation for eigenvalues",
    ])

    # Biology
    lines.extend([
        "photosynthesis converts CO2 and H2O into glucose C6H12O6 and O2 using light energy",
        "cellular respiration: C6H12O6 + 6O2 -> 6CO2 + 6H2O + 38 ATP",
        "DNA replication is semiconservative: each strand serves as template",
        "transcription: DNA -> mRNA in the nucleus by RNA polymerase",
        "translation: mRNA -> protein at ribosomes using tRNA anticodons",
        "mitosis produces two identical diploid daughter cells for growth and repair",
        "meiosis produces four haploid gametes with genetic recombination and crossing over",
        "natural selection acts on phenotypic variation in populations driving evolution",
        "Hardy-Weinberg equilibrium: p^2 + 2pq + q^2 = 1 for allele frequencies in large random mating populations",
        "enzyme kinetics: v = Vmax*[S]/(Km + [S]) Michaelis-Menten equation describes saturation",
        "the Krebs cycle (citric acid cycle) generates NADH, FADH2, and GTP in the mitochondrial matrix",
        "osmosis is the diffusion of water across a semipermeable membrane from low to high solute concentration",
        "electrochemical gradient drives ATP synthesis in chemiosmosis through ATP synthase",
        "action potential propagation: resting (-70mV) -> depolarization (+30mV) -> repolarization -> refractory period",
        "the central dogma of molecular biology: DNA -> RNA -> protein describes information flow",
        "restriction enzymes cut DNA at specific palindromic recognition sequences",
        "PCR polymerase chain reaction amplifies DNA using thermal cycling: denature anneal extend",
        "CRISPR-Cas9 is a genome editing tool that uses guide RNA to target specific DNA sequences",
        "the lac operon in E. coli is a model of gene regulation with operator promoter and structural genes",
        "population growth models: exponential dN/dt = rN and logistic dN/dt = rN(1-N/K)",
    ])

    # Chemistry concepts
    lines.extend([
        "electronegativity increases across a period and decreases down a group in the periodic table",
        "pH = -log10[H+] measures hydrogen ion concentration on a scale from 0 to 14",
        "oxidation is loss of electrons, reduction is gain of electrons (OIL RIG mnemonic)",
        "equilibrium constant Keq = [products]/[reactants] at equilibrium raised to stoichiometric powers",
        "Gibbs free energy: delta_G = delta_H - T*delta_S determines reaction spontaneity",
        "Hess's law: total enthalpy change is independent of the pathway taken",
        "Le Chatelier's principle: a system at equilibrium shifts to counteract an applied stress",
        "electron configuration of Fe: [Ar] 3d6 4s2, Fe2+: [Ar] 3d6, Fe3+: [Ar] 3d5",
        "hybridization: sp3 (tetrahedral 109.5), sp2 (trigonal planar 120), sp (linear 180)",
        "rate law: rate = k[A]^m[B]^n where m,n are experimentally determined reaction orders",
        "Nernst equation: E = E0 - (RT/nF)*ln(Q) relates cell potential to non-standard concentrations",
        "colligative properties: boiling point elevation delta_Tb = Kb*m*i depends only on solute particles",
        "bond energy: C-H 413 kJ/mol, C=C 614 kJ/mol, C-C 348 kJ/mol, O-H 463 kJ/mol",
        "the ideal gas constant R = 8.314 J/(mol*K) = 0.08206 L*atm/(mol*K)",
        "first law of thermodynamics: delta_U = q + w, internal energy change equals heat plus work",
        "Arrhenius equation: k = A*exp(-Ea/RT) relates rate constant to temperature and activation energy",
        "the Born-Haber cycle calculates lattice energy using Hess's law and measurable quantities",
        "molecular orbital theory: bonding orbitals are lower energy than antibonding orbitals",
        "crystal field theory: octahedral d-orbital splitting gives t2g (lower) and eg (upper) sets",
        "the Clausius-Clapeyron equation: ln(P2/P1) = -delta_Hvap/R * (1/T2 - 1/T1)",
    ])

    # Extended derivations
    lines.extend([
        "Deriving kinetic energy from work-energy theorem: "
        "W = integral F dx = integral ma dx = integral m(dv/dt)dx = integral mv dv = 0.5mv^2 - 0.5mv0^2. "
        "Therefore KE = 0.5mv^2.",

        "Deriving the ideal gas law from kinetic theory: "
        "pressure P = (1/3)*rho*v_rms^2 where rho = Nm/V. "
        "Average kinetic energy = (3/2)kT per molecule. "
        "So PV = NkT = nRT where R = NA*k.",

        "The fundamental theorem of calculus connects differentiation and integration: "
        "if F(x) = integral_a^x f(t)dt then F'(x) = f(x). "
        "This means integration is the inverse of differentiation. "
        "Part 2: integral_a^b f(x)dx = F(b) - F(a) where F is any antiderivative of f.",

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
        "For Earth: r_s = 2*6.674e-11*5.972e24/(3e8)^2 = 8.87 mm. "
        "For the Sun: r_s = 2*6.674e-11*1.989e30/(3e8)^2 = 2.95 km.",

        "Dimensional analysis of the period of a pendulum: "
        "T depends on length L, mass m, and gravity g. "
        "[T] = [L]^a [m]^b [g]^c -> s = m^a * kg^b * (m/s^2)^c. "
        "Solving: a=1/2, b=0, c=-1/2, so T = k*sqrt(L/g). "
        "The mass does not affect the period. Exact: T = 2*pi*sqrt(L/g).",

        "Deriving the rocket equation (Tsiolkovsky): "
        "thrust F = -v_e * dm/dt where v_e is exhaust velocity. "
        "Integrating: delta_v = v_e * ln(m_0/m_f) where m_0 is initial mass and m_f is final mass. "
        "This logarithmic relationship means exponentially more fuel for linear delta_v gains.",

        "Deriving the Euler-Lagrange equation: "
        "the action S = integral L(q, dq/dt, t) dt is stationary for the true path. "
        "Requiring delta_S = 0 gives: d/dt(partial L / partial dq/dt) - partial L / partial q = 0. "
        "For a free particle L = 0.5*m*v^2, this gives m*a = 0 as expected.",

        "Deriving the normal distribution from the central limit theorem: "
        "the sum of n independent random variables with mean mu and variance sigma^2 "
        "approaches N(n*mu, n*sigma^2) as n -> infinity. "
        "The PDF is f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). "
        "68% of data falls within 1 sigma, 95% within 2 sigma, 99.7% within 3 sigma.",

        "Deriving the heat equation from Fourier's law: "
        "heat flux q = -k * dT/dx (Fourier's law). "
        "Conservation of energy in a thin slab: rho*c_p*dT/dt = k*d^2T/dx^2. "
        "The thermal diffusivity alpha = k/(rho*c_p) gives dT/dt = alpha * d^2T/dx^2. "
        "Solutions involve separation of variables: T(x,t) = X(x)*T_func(t).",

        "Deriving Kepler's third law from Newton's gravitation: "
        "for circular orbit: G*M*m/r^2 = m*v^2/r where v = 2*pi*r/T. "
        "Substituting: G*M/r = 4*pi^2*r^2/T^2. "
        "Rearranging: T^2 = (4*pi^2/GM)*r^3. "
        "The ratio T^2/r^3 is constant for all planets orbiting the same star.",

        "The divergence theorem connects volume and surface integrals: "
        "integral_V (nabla . F) dV = integral_S F . dA "
        "where S is the closed surface bounding volume V. "
        "Physical interpretation: total flux out of V equals integral of divergence inside. "
        "Applying to electric field: integral E . dA = Q_enclosed/epsilon_0 gives Gauss's law.",

        "Proving the Pythagorean theorem algebraically: "
        "consider a square of side (a+b) containing a tilted square of side c. "
        "Area of outer square: (a+b)^2 = a^2 + 2ab + b^2. "
        "Area also equals: 4*(0.5*a*b) + c^2 = 2ab + c^2. "
        "Therefore a^2 + 2ab + b^2 = 2ab + c^2, giving a^2 + b^2 = c^2.",

        "The Boltzmann distribution describes the probability of a microstate: "
        "P(E_i) = (1/Z) * exp(-E_i / kT) where Z = sum_j exp(-E_j / kT) is the partition function. "
        "At low temperature, the system occupies the ground state. "
        "At high temperature, all states become equally likely. "
        "The average energy is <E> = -d(ln Z)/d(beta) where beta = 1/(kT).",
    ])

    return lines


def main():
    print("Building tokenizer training corpus...")
    print("  Target: 500K-1M+ characters for 8192 vocab BPE\n")

    all_texts = []

    # 1. Load local data
    local = load_jsonl_texts()
    print(f"  Local data: {len(local)} examples")
    all_texts.extend(local)

    # 2. Download real STEM data from HuggingFace (~80% of corpus)
    print("\n  Downloading real STEM data (streaming)...")
    try:
        real_texts = []
        real_texts.extend(download_openwebmath(target_chars=5_000_000))
        real_texts.extend(download_proofpile2(target_chars=5_000_000))
        real_texts.extend(download_bigmath(target_chars=5_000_000))
        real_texts.extend(download_algebraicstack(target_chars=3_000_000))
        print(f"\n  Total real data: {len(real_texts)} examples, "
              f"{sum(len(t) for t in real_texts):,} chars")
        all_texts.extend(real_texts)
    except ImportError:
        print("  WARNING: 'datasets' not installed. Run: pip install datasets")
        print("  Falling back to synthetic-only corpus.")
    except Exception as e:
        print(f"  WARNING: Download failed: {e}")
        print("  Falling back to synthetic-only corpus.")

    # 3. Synthetic STEM text (~20% of corpus, notation coverage)
    synthetic = generate_synthetic_stem()
    print(f"\n  Synthetic data: {len(synthetic)} lines")
    all_texts.extend(synthetic)

    # Write corpus
    os.makedirs("tokenizer", exist_ok=True)
    corpus_path = os.path.join("tokenizer", "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text.strip() + "\n")

    total_chars = sum(len(t) for t in all_texts)
    total_mb = total_chars / (1024 * 1024)
    print(f"\n  Corpus written to {corpus_path}")
    print(f"  Total: {len(all_texts):,} entries, {total_chars:,} chars ({total_mb:.1f} MB)")

    if total_chars < 200_000:
        print("\n  WARNING: Corpus may be too small for 8192 vocab.")
        print("  Install 'datasets' and re-run for real STEM data.")


if __name__ == "__main__":
    main()
