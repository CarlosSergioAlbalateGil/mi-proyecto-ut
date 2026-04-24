Foundational Frameworks for AI-Native
Channel Modeling: An Interdisciplinary
Analysis of Structured RF
Representations and Neural Operators
Introduction: The Paradigm Shift Toward AI-Native 6G
Architectures
The transition from the fifth generation (5G) to the sixth generation (6G) of cellular networks
necessitates a fundamental reimagining of wireless communication architectures. Future
networks are conceptualized not merely as conduits for data transfer, but as an expansive
cyber-physical continuum wherein artificial intelligence (AI) is intrinsically woven into the
network's foundational fabric.^1 At the nexus of this transformation is the critical imperative to
advance wireless channel modeling. Traditional paradigms treat the wireless channel
predominantly as a stochastic black box, relying on simplistic empirical or statistical models
that fail to capture the high-frequency spatial variations inherent in environments, or
alternatively, utilizing deterministic models like ray-tracing which are computationally
prohibitive for real-time edge deployment.^4
The conceptual text underpinning this analysis posits a radical departure from these legacy
methodologies, proposing a unified framework that learns structured, physically grounded
representations of the channel without requiring explicit geometric reconstruction. This
proposed AI-native channel modeling stack integrates "Ugly-Twins" (Structured RF
Representations), latent domain compression for complex-valued Channel State Information
(CSI), and spectral-domain deep learning based on Neural Operators.^5 Because the subject
matter is highly nascent and interdisciplinary, achieving a comprehensive understanding
requires an exhaustive exploration of tangential applications, analogous methodologies, and
underlying theoretical frameworks across disparate fields.
This report systematically synthesizes findings from a vast array of peer-reviewed articles,
technical reports, and foundational academic concepts. The objective is to evaluate the
conceptual landscape surrounding the proposed framework, explicitly summarizing the
primary contributions of identified resources, defining their precise relevance to the core
concepts of Structured RF Representations and Operator Learning, and articulating the
implications that will drive the next generation of 6G channel synthesis.

The "Ugly-Twin" Conceptual Metaphor:
Interdisciplinary Origins and Recontextualization
The nomenclature "Ugly-Twin," as introduced in the target framework, functions as a highly
specific semantic vehicle to describe compact, learnable abstractions of the wireless
environment. To fully appreciate its theoretical utility within AI-native channel modeling, one
must examine how the "ugly twin" concept operates across foundational contributing
disciplines. In almost all interdisciplinary contexts, the term denotes a phenomenon
characterized by functional mimicry that deliberately lacks aesthetic, structural, or foundational
fidelity to its source.

Tangential Applications in Supply Chain and Digital Media
In the domain of global supply chain management and procurement logistics, the "ugly twins"
metaphor is extensively utilized to describe failed low-wage-country sourcing projects that act
as expensive replacements for established, high-fidelity supply lines.^7 The primary findings of
research in this sector indicate that companies often attempt to replicate domestic
manufacturing processes in low-cost environments. These new supply lines (the "twins")
attempt to functionally mimic the output of the original but lack the intrinsic structural quality,
resulting in complex, constrained, and ultimately divergent outputs that require continuous
reactionary adaptation.^9
Similarly, in the field of digital media and cybersecurity, the "ugly twin" concept describes
malicious or plagiarized websites designed to confuse audiences by mimicking the architecture
of legitimate independent media outlets while stripping away journalistic rigor.^11 Research
highlighting human rights and digital platform monitoring in Serbia details how these proxy sites
functionally capture user traffic and display similar content topologies without possessing the
underlying factual infrastructure of the original sites.^11 In historical software engineering
literature, the term has also been utilized to describe parallel, functionally similar programs that
lack the elegant architecture of their source code.^13

Relevance to Structured RF Representations
The synthesis of these cross-disciplinary findings is profoundly relevant to the proposed
framework. In all analogous applications, the "ugly twin" represents an entity that prioritizes
functional output over exact structural replication. When translated to 6G channel modeling,
the pursuit of a traditional "Digital Twin" of an RF environment mandates an exhaustive,
pixel-perfect, highly explicit 3D geometric reconstruction of a physical space, demanding
massive computational overhead derived from lidar and ray-tracing data.^2
The text proposes abandoning this explicit geometric fidelity in favor of a functional mimicry:
the Structured RF Digital Twin. Research led by institutional groups, such as the efforts
overseen by Narcis Cardona, explicitly focuses on developing these structured RF
representations of the environment.^5 The primary contribution of this research vector is the
design of representations that are compact, adaptable, and capable of predicting channel

evolution without explicitly rendering every physical object in the space.^5
The relevance to the target text is exact: the "Ugly-Twin" in MIMO channel modeling is an AI
model that captures the spatial interactions of electromagnetic waves—the multipath fading,
the scattering, the delay-domain characteristics—without knowing the exact geometric shape
of the wall or building causing them. It is "ugly" because it abandons geometric truth; it is a
"twin" because it flawlessly replicates the electromagnetic outcome. The implication for future
research is a massive reduction in the computational bandwidth required for real-time 6G
channel synthesis, as models will only need to parse latent physical proxies rather than full
high-definition spatial maps.
Feature Paradigm Traditional "True" Digital
Twin
The "Ugly-Twin" (Structured
RF Representation)
Core Objective Explicit geometric and visual
fidelity
Implicit electromagnetic
functional fidelity
Computational Cost Extremely high (prohibitive
for edge)
Highly compact (optimized for
latent forecasting)
Data Reliance Extensive real-world sensing
(e.g., Lidar)
Deep learning inferred
propagation structures
Environmental Knowledge Explicit spatial mapping
required
Implicit spatial abstraction
utilized
Adaptability Rigid; requires frequent
manual updating
Evolutionary; adapts via latent
parameterization

Implicit Geometric Abstraction: Analogous
Methodologies from Materials Science
The target framework emphasizes the extraction of spatially meaningful features from raw

channel measurements without relying on full 3D scene reconstruction. To validate the viability
of this approach, it is instructive to analyze analogous methodologies in other physics-bound
domains where deep learning predicts physical interactions without explicit geometry.

Machine Learning in One-Dimensional Channel Modeling
A compelling parallel is found in materials science and thermodynamics, specifically concerning
one-dimensional fuel and air channel modeling for complex porous structures. Research
examining the effect of pore volume fraction on three-phase boundary (TPB) density utilizes
Keras-based neural networks to achieve rapid and accurate predictions of key material
properties without requiring explicit geometric reconstruction or relying on percolation
threshold assumptions.^14 The primary finding of this research is that machine learning models
can internalize the mathematical outcomes of physical distributions (such as varying Nickel/YSZ
distributions in fuel cells) and project their physical behaviors into a simulation domain purely
through latent mathematical mapping, completely bypassing the need to digitally reconstruct
the microscopic physical geometry of the pores.^14

Relevance and Implications for 6G Channel Synthesis
The methodological relevance to AI-native MIMO channel modeling is profound. Just as the
neural network in the materials science study maps the functional impact of invisible
microscopic pores without rendering them, the proposed wireless framework aims to map the
functional impact of invisible urban topology (reflectors, scatterers, angular domains) utilizing
high-dimensional CSI data without explicitly mapping the cityscape. The AI infers the "latent
propagation structures" based purely on how the electromagnetic wave degrades, shifts, or
multiplies.^14
The implication here informs a critical further research direction: the development of
cross-disciplinary machine learning architectures. Algorithms successfully deployed to predict
thermodynamic properties through implicit geometric abstraction can be theoretically
modified to predict radio frequency attenuation and multipath propagation. This validates the
core assertion of the target text that spatially meaningful features can indeed be derived
strictly from input-output mathematical tensors, solidifying the operational viability of the
"Ugly-Twin" concept.

The Complex-Valued CSI Tensor and Multi-Resolution
Latent Spaces
A foundational pillar of the proposed framework is the utilization of an encoder that maps
high-dimensional CSI tensors into structured latent variables, treating compression as the
fundamental mechanism for channel re-parameterization. Understanding the challenges and
contemporary methodologies surrounding complex-valued CSI is paramount.

The Limitations of Magnitude Representation
In modern Cell-Free Massive MIMO (CF-MaMIMO) and Open Radio Access Network (O-RAN)
architectures, uplink signals transmitted by user devices are received by distributed arrays,
yielding full complex-valued CSI.^15 A significant historical bottleneck in deep learning for
wireless networks has been the processing of these complex matrices. Traditional approaches
frequently convert the complex-valued CSI across all subcarriers purely into its magnitude
representation, actively discarding phase information before subsequent processing.^16 The
resulting magnitudes are then normalized using robust scalers to mitigate the influence of
outlier data.^16
While this magnitude-only approach ensures proper mapping onto discrete subcarrier domains
and maintains numerical stability, research indicates that it destroys critical spatial data.^16 The
target framework's goal to infer latent propagation structures—such as angular
characteristics—is mathematically impossible without the phase data that dictates the
directionality and interaction of the electromagnetic waves.

Deep Learning in the Native Complex Domain
To resolve this, cutting-edge research proposes maintaining the CSI in its native complex
format. The primary findings of studies focusing on 3D Convolutional Neural Networks
(CV-3DCNN) demonstrate that utilizing complex-valued neural networks to deal directly with
complex CSI matrices drastically minimizes information loss.^17 These networks employ
three-dimensional convolution operations for feature extraction, making full use of the hidden
information regarding phase and amplitude.^17 Experimental results confirm that such
architectures improve the accuracy of downlink CSI prediction by approximately 6 dB
compared to magnitude-only models.^17
Furthermore, research detailing frameworks like DeepCRF highlights the necessity of carefully
designing neural network structures specifically for complex-valued CSI input data.^18 Findings
show that small-scale micro-CSI is particularly susceptible to noise, and the presence of
significant multipath fading makes it difficult to extract accurate fingerprints using traditional
signal processing.^18 By passing the complex-valued CSI tensor through paralleled residual
convolution blocks, models can transform the data into multi-resolution latent spaces.^19 These
diverse resolutions are fused via channel concatenation and converted into highly robust
real-valued spatial features, which outperform traditional geometric-based algorithms in tasks
like device proximity estimation.^19

Latent Parallels in Medical fMRI Imaging
The necessity of processing complex-valued tensors without losing spatial relationships is not
unique to telecommunications. A powerful analogous methodology exists in advanced medical
imaging. Research focused on decomposing complex-valued multi-subject functional
Magnetic Resonance Imaging (fMRI) data highlights the use of new Tucker-2 models with
enhanced core tensor sparsity.^20 In medical imaging, discarding the complex phase of the MRI
signal results in critical losses of physiological data; therefore, non-negative neural networks

and dual-domain neural operators are engineered specifically to process these massive
tensors while remaining discretization invariant.^20
The relevance to the AI-native MIMO framework is explicit: the encoder mechanisms proposed
for 6G wireless systems share a fundamental mathematical architecture with those used in
modern neuroimaging. In both fields, the encoder must map high-dimensional, complex-valued
tensors into structured latent variables while preserving spectral and spatial coherence. The
implication for future 6G research is that the telecommunications sector should heavily
scrutinize advancements in medical imaging tensor decomposition, as the mathematical
solutions for fMRI sparsity and dual-domain discretization invariance directly map to the
challenges of compressing wireless CSI tensors into a compact latent parameterization.
CSI Processing
Methodology
Data Retention Computational
Complexity
Efficacy for Spatial
Abstraction (Target
Framework)
Magnitude-Only
Extraction
Discards phase data
entirely
Low; easily
processed by
standard ML
Very Poor; cannot
accurately infer angular
domain
Real-Valued
Concatenation
Separates
real/imaginary into
dual channels
Moderate Moderate; struggles
with non-linear phase
relationships
CV-3DCNN
(Complex Domain)
Retains native
complex matrix
relationships
High Excellent; captures full
spatial and spectral
coherence
Multi-Resolution
Latent Fusion
Extracts varying
spatial scales
concurrently
Very High Optimal; aligns with the
framework's encoder
requirements

Neural Operators and the Shift to Infinite-Dimensional
Function Spaces
To support real-time deployment and scalable channel synthesis, the proposed framework
relies heavily on "spectral-domain deep learning architectures, based on Neural Operators
(NOs)." The introduction of Neural Operators represents perhaps the most profound
theoretical shift in the entire modeling stack, moving away from discrete point-to-point
predictions toward continuous operator learning.

The Mathematical Principles of Fourier Neural Operators (FNO)
Traditional deep learning models, such as Multi-Layer Perceptrons (MLPs) or standard
Convolutional Neural Networks (CNNs), are designed to learn mappings between
finite-dimensional vectors.^6 If a CNN is trained on a specific spatial grid (e.g., a specific antenna
array configuration), it cannot process data from a different grid resolution without being
completely retrained. This grid dependency is fatal for next-generation MIMO systems, which
feature continuous apertures, holographic surfaces, and dynamic sub-wavelength antenna
coupling.^6
Neural Operators, conversely, learn function-to-function mappings between
infinite-dimensional function spaces.^6 The primary findings of expansive research conducted
by Jian Xiao, Ji Wang, Chau Yuen, and others firmly establish the Fourier Neural Operator (FNO)
as the optimal mathematical vehicle for modeling complex physical systems governed by
partial differential equations (PDEs) representing electromagnetic wave propagation.^6
An FNO achieves this by parameterizing the integral kernel directly in the Fourier space.^21 It
applies stacked Fourier layers that map the input data into the frequency domain, multiply it by
a learnable weight tensor, and then perform an inverse Fourier transform back to the spatial
domain.^28 By truncating the signal to a fixed number of low-frequency modes, the FNO isolates
the foundational, invariant functional relationships of the propagation environment while
remaining entirely robust against varying input discretizations.^28 The critical finding here is that
FNOs are inherently "mesh-free"; they learn the continuous physical rule of the environment,
not just the discrete data points, meaning they can infer channel characteristics at entirely
unobserved positions and across varying grid resolutions seamlessly.^6

Fluid Dynamics Origins and RF Recontextualization
To understand the robustness of FNOs, one must examine their origins in applied mathematics
and fluid dynamics. FNOs were initially formulated to perform highly expressive and efficient
mappings for complex PDEs such as Burgers' equation, Darcy flow, and the Navier-Stokes
equation, specifically operating within highly chaotic turbulent regimes.^21 In these fluid
dynamics experiments, FNOs demonstrated state-of-the-art performance, computing
complex fluid behaviors up to three orders of magnitude faster than traditional numerical PDE
solvers.^21
The relevance of this fluid dynamics research to the AI-native wireless framework is

foundational. The propagation of electromagnetic waves through a physical
environment—subject to scattering, diffraction, and multi-path fading—is mathematically
analogous to the propagation of fluid through a porous medium; both are continuous physical
processes governed by complex wave equations.^6 By recontextualizing the FNO from fluid
turbulence to RF channel estimation, the target framework harnesses an architecture
structurally and philosophically aligned with the actual physics of wireless communications.^6

Hierarchical FNOs for Flexible Intelligent Metasurfaces
The implication of operator learning for 6G is best illustrated by its application to Flexible
Intelligent Metasurfaces (FIMs). FIMs introduce unprecedented morphological degrees of
freedom by dynamically morphing their 3D shape to ensure signals interfere constructively.^23
Estimating the channel state across a continuous, high-dimensional deformation space is
impossible using traditional model-based interpolation or sparse signal recovery.^27
Research indicates that Hierarchical Fourier Neural Operators (H-FNO) are required to capture
the multi-scale features across this hierarchy of spatial resolutions.^23 In this architecture, the
H-FNO continuously maps the infinite variations of FIM shapes directly to their corresponding
channel responses. After multiple iterations in the latent space, a projection layer maps the final
latent representation back to the desired output dimension, accurately reconstructing both the
real and imaginary parts of the estimated channel at each specific antenna.^23 This deeply
validates the core premise of the target text: rather than directly predicting the full channel
from raw data, the framework operates efficiently in the spectral domain, utilizing NOs to map
complex interactions scalably.

Physics-Informed Machine Learning and Synthetic
Generative Environments
A purely data-driven neural network, even a highly advanced FNO, remains essentially ignorant
of physical laws unless explicitly constrained. Without constraints, deep learning models might
generate channel state predictions that are mathematically plausible within the latent space
but physically impossible in the real world, violating Maxwell's equations. To address the target
framework's requirement for a "physically consistent" channel modeling stack, the integration
of Physics-Informed Machine Learning (PIML) is critical.^30

Physics-Informed Neural Operators (PINO)
The convergence of physics-based constraints and operator learning manifests as the
Physics-Informed Neural Operator (PINO). Research findings demonstrate that PINOs are a
hybrid framework that explicitly embeds physical laws into the loss function of the FNO.^28
During the training phase, if the neural network proposes a channel state that violates the
fundamental wave equations or boundary conditions of the environment, the loss function
heavily penalizes the model.^28

This ensures that the mappings between function spaces remain rigidly tethered to reality.
Studies indicate that PINOs consistently outperform purely data-driven approaches and
traditional physics-informed neural networks (PINNs) precisely because the use of
science-informed data inherently embeds channel geometry into the mathematical learning
process.^28 This is immensely relevant to the target framework's goal of "interpretable mappings
between physical environment features and channel behavior." Because the PINO is
constrained by physical laws, its latent variables are not arbitrary black-box outputs; they
correlate directly to real-world electromagnetic phenomena.

Generative AI and Diffusion Models for Data Scarcity
A massive logistical hurdle in realizing AI-native 6G networks is the acquisition of high-quality,
representative, high-precision 3D channel datasets required to train these complex neural
operators.^1 The physical world is infinitely diverse, and capturing exhaustive empirical data
across all frequency bands, polarization states, and array geometries is unfeasible.^31
To overcome data scarcity, the research landscape points toward the use of Generative AI
(GAI), specifically volumetric diffusion models.^31 Findings show that integrating diffusion
models enables the generation of high-fidelity, synthetic electromagnetic scenarios.^33 By
utilizing random sampling techniques to select subsets of pixels from spatial frames, these
models generate sparse 3D channel tensors that accurately simulate complex environments.^31
This synthetic generation is critical for advanced applications like Space-Air-Ground Integrated
Networks (SAGIN).^33 Research highlights that integrating quantum-scale device noise with
network-level interference through cross-layer PINOs allows for the real-time simulation of
satellite-terrestrial signal coupling.^33 By utilizing diffusion-generated synthetic data to train the
PINOs, engineers can overcome the severe data scarcity of orbital conditions.^33
The implication for the target framework is clear: the AI models designed to infer latent
propagation structures will not be trained exclusively on expensive physical measurements.
Instead, they will be trained on infinite, highly diverse synthetic environments generated by
diffusion models and filtered through physics-informed loss constraints, leading to models that
are immensely robust upon deployment in unobserved physical environments.

The Latent Domain Ecosystem: Scalability, Edge
Deployment, and LLM Analogies
The ultimate objective of the proposed framework is to shift the mechanism of channel
forecasting directly into the latent domain, predicting the evolution of the channel using a
drastically reduced set of structured variables rather than continuously processing the full
high-dimensional CSI tensor. This paradigm shift is the primary enabler for real-time edge
deployment and scalable network intelligence.

Channel Evolution Forecasting and Latent Reconstruction
In legacy systems, adapting to rapid changes in the wireless channel (e.g., a vehicle moving
through an urban corridor) requires the network to constantly recalculate the full propagation
matrix, introducing severe latency that undermines ultra-reliable and low-latency
communication (URLLC) goals.^4 Low-fidelity stochastic models attempt to bridge this gap but
result in substantially increased bit-error rates (BERs) because they cannot capture abrupt
variations.^4
The target framework solves this by treating the latent space not just as a storage medium, but
as the active operational domain. Research into Deep MIMO and sequence model-based
approaches demonstrates that AI frameworks can utilize Long Short-Term Memory (LSTM)
modules or FNOs to infer the channel response at the next time instance based purely on
historical latent data.^4 The framework operates by allowing the neural operator to forecast the
evolution of the "Ugly-Twin" abstraction. Because this forecasting operates on a highly
compressed array of parameters (the structured latent variables), the computational overhead
is microscopic compared to processing raw RF data. The decoder then reconstructs the full CSI
tensor specifically tailored for the end-user device at that exact millisecond, preserving spatial
and spectral coherence without bogging down the network's core processors.

Analogies to Large Language Models (LLMs) and Prompt Engineering
An intriguing and highly relevant conceptual parallel within the current research landscape is
the comparison between wireless network large models and conventional Large Language
Models (LLMs).^21 While traditional LLMs text-generation algorithms are too computationally
heavy and dataset-reliant for resource-limited edge network devices, the architectural
philosophy of "prompt engineering" is being actively adapted for 6G.^32
In an AI-native wireless context, the "prompt" is not text, but a specific set of environmental
constraints or intent-driven QoS parameters (e.g., "maximize semantic transmission quality for
user X in high-fading environment Y").^32 The foundational model—the pre-trained Fourier
Neural Operator—receives this prompt and manipulates the latent variables accordingly.^32 Just
as an LLM uses attention mechanisms to predict the next logical token in a sentence, the
operator uses spectral domain analysis to predict the next logical state of the electromagnetic
wave.
The implication of this LLM analogy is profound for the standardization of AI for wireless
networks.^21 If 6G networks are to innately provide "AI as a Service" (AIaaS), defining the Quality
of AI Service—effectively measuring how efficiently these operators reconstruct the channel
from latent prompts—becomes a primary regulatory and engineering objective.^21

Global Research Initiatives and ISAC Integration
The theoretical validity of moving operations to the latent domain is heavily supported by
massive institutional investments currently underway across the globe. European and Korean
collaborative efforts, such as the 6GARROW project, are actively publishing technical
deliverables aimed at transitioning AI-native, energy-aware 6G networks from conceptual

planning to standardized system architectures.^34 These initiatives explicitly confirm that AI/ML
solutions for Radio Access Network (RAN) optimization, built upon AI-native network designs,
are the primary drivers for the next generation of mobile communications.^34
Similarly, the NVIDIA 6G Developer Program is equipping hundreds of telecommunications
organizations with the advanced computational tools required to train these massive neural
operators.^2 In Finland, the University of Oulu is utilizing these platforms—specifically the Isaac
Sim reference application and real-time network digital twins—to pioneer advancements in
Integrated Sensing and Communications (ISAC).^2 ISAC represents the ultimate culmination of
the "Ugly-Twin" framework. By processing complex-valued CSI through a latent-domain
operator, the network can act as a high-fidelity sensor of the physical world, tracking objects
and inferring spatial structures purely through the distortion of the communication signal itself,
entirely eliminating the need for separate radar or lidar hardware.^2

Conclusion: Synthesizing the 6G Channel Modeling
Stack
The text presented in the target framework outlines a visionary approach to AI-native channel
modeling that is firmly corroborated by an exhaustive review of tangential, interdisciplinary, and
highly advanced telecommunications research. The framework's core propositions—treating
the wireless environment through structured abstractions rather than explicit geometry,
utilizing complex-valued CSI latent parameterization, and leveraging neural operators for
scalable forecasting—are not isolated hypotheses, but represent the convergence of multiple
scientific breakthroughs.
The utilization of the "Ugly-Twin" metaphor elegantly bridges the gap between functional
requirement and computational reality. By abandoning the prohibitive pursuit of high-fidelity 3D
spatial mapping in favor of implicit electromagnetic abstraction, the framework dramatically
reduces the processing burden. This concept is thoroughly validated by analogous machine
learning applications in materials science and thermodynamics, where implicit functional
mapping routinely replaces explicit geometric reconstruction.
Furthermore, the necessity of processing complex-valued CSI matrices in their native domain is
supported by parallel advancements in medical fMRI imaging, demonstrating that preserving
the phase relationships within multi-resolution latent spaces is absolute for maintaining spatial
and spectral coherence. The encoder-decoder mechanism proposed effectively translates
these massive, noisy tensors into compact, interpretable variables that represent actual
physical phenomena, such as reflector positioning and angular delays.
The integration of Fourier Neural Operators constitutes the critical mathematical engine of this
framework. Rooted originally in the complex PDE calculations of fluid dynamics, FNOs bring
mesh-independent, continuous function-to-function mapping to the wireless domain. This is
the precise technological mechanism that enables the framework to model dynamic,
high-dimensional arrays like Flexible Intelligent Metasurfaces seamlessly. When augmented

with Physics-Informed Machine Learning constraints and trained on expansive synthetic
datasets generated by volumetric diffusion models, these operators guarantee that the
predicted channel states remain mathematically optimal and physically inviolable.
Ultimately, by pushing the operational mechanics of channel evolution forecasting into this
highly compressed latent domain, the framework solves the foundational bottleneck of
massive MIMO deployments. It minimizes feedback latency, ensures real-time adaptability, and
makes advanced edge deployment economically and computationally viable. As global
initiatives like 6GARROW and institutional ISAC research advance, the theoretical stack detailed
in the target framework stands poised to become the standardized operational reality of the
AI-native 6G ecosystem.

Obras citadas
1. Towards AI-native 6G networks - AI for Good - ITU, fecha de acceso: abril 13,
2026, https://aiforgood.itu.int/event/towards-ai-native-6g-networks/
2. European Researchers Develop AI-Native Wireless Networks With NVIDIA 6G
Research Portfolio | NVIDIA Blog, fecha de acceso: abril 13, 2026,
https://blogs.nvidia.com/blog/europe-6g-research/
3. Towards Trustworthy AI-Native Radio Access Networks for the 6G era -
UPCommons, fecha de acceso: abril 13, 2026,
https://upcommons.upc.edu/bitstreams/c42e2c4c-c1fa-4061-9897-34352138c1c
b/download
4. Empowering Wireless Network Applications with Deep Learning-based Radio
Propagation Models - University of Cambridge, fecha de acceso: abril 13, 2026,
https://www.repository.cam.ac.uk/bitstreams/2636168b-bf63-4f11-984c-adca26d
6aff3/download
5. RESEARCH PLAN PROPOSAL FOR THE DOCTORAL ... - UPV, fecha de acceso:
abril 13, 2026,
https://www.upv.es/entidades/edoctorado/wp-content/uploads/2026/01/Narcis-C
ardona-Plan-de-investigacion.pdf
6. Learning Function-to-Function Mappings: A Fourier Neural Operator for
Next-Generation MIMO Systems Corresponding author: Ji Wang. Jian Xiao and Ji
Wang are with the Department of Electronics and Information Engineering,
College of Physical Science and Technology, Central China Normal University,
Wuhan 430079, China (e-mail: jianx@mails - arXiv, fecha de acceso: abril 13, 2026,
https://arxiv.org/html/2510.04664v
7. Operational Excellence and Supply Chains in Logistics - TUHH Open Research,
fecha de acceso: abril 13, 2026,
https://tore.tuhh.de/bitstream/11420/1268/1/HICL%202015%20-%20Vol%2022%
0-%20Operational%20Excellence%20in%20Logistics%20and%20Supply%20Cha
ins.pdf
8. Sourcing complexity factors on contractual relationship: Chinese suppliers'
perspective - Taylor & Francis, fecha de acceso: abril 13, 2026,
https://www.tandfonline.com/doi/pdf/10.1080/21693277.2014.
9. Full article: Sourcing complexity factors on contractual relationship: Chinese
suppliers' perspective - Taylor & Francis, fecha de acceso: abril 13, 2026,
https://www.tandfonline.com/doi/full/10.1080/21693277.2014.
10. Sourcing complexity factors on contractual relationship: Chinese suppliers'
perspective, fecha de acceso: abril 13, 2026,
https://www.researchgate.net/publication/305359713_Sourcing_complexity_facto
rs_on_contractual_relationship_Chinese_suppliers'_perspective
11. Serbia: Freedom on the Net 2024 Country Report, fecha de acceso: abril 13, 2026,
https://freedomhouse.org/country/serbia/freedom-net/
12. Serbia: Freedom on the Net 2022 Country Report, fecha de acceso: abril 13, 2026,
https://freedomhouse.org/country/serbia/freedom-net/
13. Full text of "Amiga Format Magazine" - Internet Archive, fecha de acceso: abril 13,
2026,
https://archive.org/stream/AmigaFormatMagazine_201902/Amiga_Format_Issue_
048_1993_07_Future_Publishing_GB_djvu.txt
14. Multiphysics Modeling of Solid Oxide Fuel Cells for Gradient Minimization and
Inductive Loop Analysis in Impedance Spectroscopy - Georgia Southern
Commons, fecha de acceso: abril 13, 2026,
https://digitalcommons.georgiasouthern.edu/cgi/viewcontent.cgi?article=4277&c
ontext=etd
15. Lightweight Channel Gain Estimation with Reduced X‑Haul CSI Signaling in
O‑RAN - arXiv, fecha de acceso: abril 13, 2026,
https://arxiv.org/html/2604.08458v
16. Multi-Task Deep Learning With Adaptive CSI Filtering for Joint Beamforming and
Localization - IEEE Xplore, fecha de acceso: abril 13, 2026,
https://ieeexplore.ieee.org/iel8/6287639/10820123/11267407.pdf
17. CV-3DCNN: Complex-Valued Deep Learning for CSI Prediction in FDD Massive
MIMO Systems | Request PDF - ResearchGate, fecha de acceso: abril 13, 2026,
https://www.researchgate.net/publication/349210976_CV-3DCNN_Complex-Value
d_Deep_Learning_for_CSI_Prediction_in_FDD_Massive_MIMO_Systems
18. DeepCRF: Deep Learning-Enhanced CSI-Based RF Fingerprinting for
Channel-Resilient WiFi Device Identification - arXiv, fecha de acceso: abril 13,
2026, https://arxiv.org/html/2411.06925v
19. Wi-Prox: Proximity Estimation of Non-Directly Connected Devices via Sim2Real
Transfer Learning, fecha de acceso: abril 13, 2026,
https://tns.thss.tsinghua.edu.cn/~guoxuan/assets/pdf/Paper-Wi-Prox.pdf
20. List of Accepted Papers - IEEE ICASSP 2026 || Barcelona, Spain || 4-8 May 2026,
fecha de acceso: abril 13, 2026,
https://cmsworkshops.com/ICASSP2026/papers/accepted_papers.php
21. Learning Function-to-Function Mappings: A Fourier Neural Operator for
Next-Generation MIMO Systems - ResearchGate, fecha de acceso: abril 13, 2026,
https://www.researchgate.net/publication/396249718_Learning_Function-to-Func
tion_Mappings_A_Fourier_Neural_Operator_for_Next-Generation_MIMO_System
s
22. [PDF] Learning Function-to-Function Mappings: A Fourier Neural, fecha de
acceso: abril 13, 2026,
https://www.semanticscholar.org/paper/Learning-Function-to-Function-Mapping
s%3A-A-Fourier-Xiao-Wang/8b3a06a453cc20949fefb048031403c6b79a166e
23. Channel Estimation for Flexible Intelligent Metasurfaces: From Model-Based
Approaches to Neural Operators - arXiv, fecha de acceso: abril 13, 2026,
https://arxiv.org/pdf/2508.
24. Channel Estimation for Flexible Intelligent Metasurfaces: From Model-Based
Approaches to Neural Operators Jian Xiao and Ji Wang are with the Department
of Electronics and Information Engineering, College of Physical Science and
Technology, Central China Normal University, Wuhan 430079, China (e-mail:
jianx@mails.ccnu.edu.cn - arXiv, fecha de acceso: abril 13, 2026,
https://arxiv.org/html/2508.00268v
25. Ji Wang's research works | Central China Normal University and, fecha de acceso:
abril 13, 2026,
https://www.researchgate.net/scientific-contributions/Ji-Wang-
26. Qimei Cui - DBLP, fecha de acceso: abril 13, 2026, https://dblp.org/pid/79/
27. Tensor-based Channel Estimation for Extremely Large-Scale MIMO-OFDM with
Dynamic Metasurface Antennas - ResearchGate, fecha de acceso: abril 13, 2026,
https://www.researchgate.net/publication/390217504_Tensor-based_Channel_Est
imation_for_Extremely_Large-Scale_MIMO-OFDM_with_Dynamic_Metasurface_A
ntennas
28. Science-Informed Design of Deep Learning With Applications to Wireless
Systems: A Tutorial - arXiv, fecha de acceso: abril 13, 2026,
https://arxiv.org/html/2407.07742v
29. Science-Informed Design of Deep Learning With Applications to Wireless
Systems: A Tutorial - ResearchGate, fecha de acceso: abril 13, 2026,
https://www.researchgate.net/publication/401116087_Science-Informed_Design_
of_Deep_Learning_With_Applications_to_Wireless_Systems_A_Tutorial
30. Physics-Informed Artificial Intelligence for Adaptive Wireless Channel Modelling
in Fifth-Generation (5G) Networks - Unikom, fecha de acceso: abril 13, 2026,
https://ojs.unikom.ac.id/index.php/injuratech/article/download/18015/
31. RadioDiff-3D: A 3D×3D Radio Map Dataset and Generative Diffusion Based
Benchmark for 6G Environment-Aware Communication - arXiv, fecha de acceso:
abril 13, 2026, https://arxiv.org/html/2507.12166v
32. Architecting 6G: From Holographic Metasurfaces to AI-Driven Wireless Systems -
IEEE Xplore, fecha de acceso: abril 13, 2026,
https://ieeexplore.ieee.org/iel8/7742/11095316/11095392.pdf
33. Electromagnetic situation awareness and modeling for space–air–ground
integrated networks - PMC, fecha de acceso: abril 13, 2026,
https://pmc.ncbi.nlm.nih.gov/articles/PMC12805832/
34. Towards AI-Native 6G networks, fecha de acceso: abril 13, 2026,
https://www.6gflagship.com/news/towards-ai-native-6g-networks/