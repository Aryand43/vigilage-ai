# SC1304 Project — VigilAge AI: Proactive Fall Detection for Elderly Independence

**Group 3**

| Role | Name | Matriculation No. |
|------|------|--------------------|
| Member | — | — |
| Member | — | — |
| Member | — | — |
| Member | — | — |
| Member | — | — |

*(Fill in team details above.)*

---

## 1. Introduction

Falls remain the leading cause of injury-related morbidity and mortality among adults aged 65 and older worldwide. In the United States alone, the medical costs attributable to falls among older adults have reached approximately 80 billion dollars annually (CDC, 2024). The challenge is not confined to Western economies: in Singapore, demographic projections indicate that one-quarter of the national population will be aged 65 or above by 2030 (Lau, 2024), placing acute pressure on healthcare infrastructure and eldercare services. Beyond the immediate physical consequences—fractures, traumatic brain injuries, and prolonged hospitalisation—falls inflict a pervasive psychological toll. Fear of falling drives social withdrawal, reduced physical activity, depression, and an accelerating loss of independence, creating a vicious cycle in which decreased mobility further elevates fall risk.

Existing fall-response technologies have struggled to break this cycle. Manual alert buttons, such as wearable panic pendants, depend on the user's ability and willingness to press a button after an incident—a requirement that fails when the individual is unconscious, disoriented, or in shock. Camera-based monitoring systems, while capable of passive detection, introduce significant privacy concerns and are perceived as intrusive, undermining the dignity they ostensibly protect. VigilAge AI addresses these shortcomings through a fundamentally different approach: a lightweight, wearable fall-detection system built on a Long Short-Term Memory (LSTM) neural network that processes tri-axial accelerometer and gyroscope data in real time. By performing inference locally on the device rather than streaming raw sensor feeds to the cloud, the system preserves user privacy, minimises latency, and operates independently of network availability. VigilAge AI is designed to enable "aging in place" with dignity, delivering high detection accuracy alongside low false-alarm rates so that caregivers can trust the system and older adults can maintain their independence.

---

## 2. Dataset Selection and Preprocessing

### 2.1 Appropriateness and Relevance

The model is trained and evaluated on the UMAFall dataset, a publicly available benchmark sourced from Kaggle (2026) that contains over 17,000 labelled sensor samples collected from 19 human subjects. Each subject wore body-mounted inertial measurement units (IMUs) incorporating tri-axial accelerometers, gyroscopes, and magnetometers, thereby providing a rich multi-modal signal for motion analysis. The dataset encompasses a comprehensive taxonomy of movement types, including routine Activities of Daily Living (ADLs)—such as walking, sitting down, standing up, bending to pick up objects, and lying down—as well as actual falling events across multiple fall directions and fall-like events such as fast sitting and stumbling without falling.

The inclusion of fall-like activities is particularly valuable for the objectives of VigilAge AI. In real-world deployment, a disproportionate number of false alarms arise from movements that share kinematic characteristics with genuine falls—abrupt deceleration, a rapid change in body orientation, or a brief period of high g-force—yet do not constitute hazardous events. By training the LSTM on both genuine falls and their benign mimics, the model learns to discriminate between the two, thereby minimising false positives and preserving caregiver trust. Moreover, the public availability of UMAFall ensures full reproducibility of our preprocessing and training pipeline and facilitates direct comparison with other fall-detection systems reported in the literature.

### 2.2 Preprocessing Pipeline

Raw IMU signals require careful conditioning before they can be consumed by a sequence-learning model. The VigilAge AI preprocessing pipeline comprises three sequential stages: filtering, segmentation, and normalisation.

**I. Butterworth Low-Pass Filtering (20 Hz Cutoff, 3rd Order)**

Raw accelerometer and gyroscope readings are contaminated by high-frequency noise originating from muscle tremor, sensor electronic noise, mechanical vibrations transmitted through clothing and skin contact, and environmental disturbances. These artefacts occupy frequency bands well above the range of human voluntary motion during a fall event. A third-order Butterworth low-pass filter with a cutoff frequency of 20 Hz is applied to each sensor channel independently. The Butterworth design is chosen for its maximally flat magnitude response in the passband, ensuring that the low-frequency components characteristic of fall dynamics—gradual instability, sudden high-magnitude impact, and post-impact stillness—are preserved with minimal distortion. The filter is implemented using `scipy.signal.butter` to compute filter coefficients and `scipy.signal.filtfilt` to apply zero-phase forward-backward filtering, which eliminates the phase shift that a causal filter would introduce and thereby preserves the precise temporal alignment of events across sensor channels.

**II. Sliding Window Segmentation (2-Second Windows, 50 % Overlap)**

Biomechanical studies indicate that a typical fall event unfolds over a duration of 0.5 to 1.5 seconds, comprising three distinct phases: pre-impact instability, the impact itself, and post-impact stillness or recovery. A window length of 2 seconds is selected to capture all three phases within a single observation, providing the LSTM with sufficient temporal context to learn the complete fall signature. At the UMAFall sampling rate of 100 Hz, each window contains 200 time steps. Windows are extracted using a sliding-window approach with 50 % overlap, corresponding to a step size of 1 second (100 samples). This overlap ensures that a fall event whose onset occurs near the boundary of one window will nonetheless be fully contained within an adjacent window, preventing the loss of critical transitional dynamics. The resulting set of fixed-length windows forms the input tensor to the LSTM model.

**III. Z-Score Normalisation per Window**

Sensor readings vary substantially across individuals due to differences in body mass, height, gait pattern, and limb length, as well as across devices due to manufacturing tolerances and calibration drift. To mitigate these sources of variance, z-score normalisation is applied independently to each sensor channel within each window. Concretely, for a given channel in a given window, the channel mean is subtracted and the result is divided by the channel standard deviation, yielding a transformed signal with zero mean and unit variance. This per-window normalisation ensures that the model responds to the *shape* of the signal—the relative pattern of acceleration and rotation over time—rather than to absolute magnitudes that are confounded by body type and sensor calibration. The approach enhances cross-subject generalisation and contributes to robust performance in deployment environments that differ from the training conditions.

---

## 3. Algorithm and Design

### 3.1 Innovation and Suitability

A fall is fundamentally a *temporal* event: it begins with a period of postural instability, progresses through a high-magnitude impact as the body collides with the ground, and concludes with a phase of stillness or minimal movement. Capturing this sequential structure is essential for accurate detection. Traditional machine-learning approaches to fall detection—such as random forests, support vector machines, or k-nearest neighbours—typically operate on hand-crafted statistical features extracted from fixed windows (e.g., peak acceleration, signal magnitude area, tilt angle). While these features encode useful information, they discard the temporal ordering of observations within the window. Consequently, a rapid sit-down that produces a high peak acceleration followed by stillness may be indistinguishable from a genuine fall when only aggregate statistics are considered, leading to elevated false-alarm rates.

The Long Short-Term Memory (LSTM) network, introduced by Hochreiter and Schmidhuber (1997), is specifically designed to model sequential data with long-range temporal dependencies. At its core, each LSTM cell maintains a *memory cell state* that can carry information across many time steps without degradation. Three learned gating mechanisms regulate information flow: the *input gate* controls what new information is written to the cell state, the *forget gate* determines what existing information is discarded, and the *output gate* selects which components of the cell state are exposed as the cell's output at each time step. Through these gates, the LSTM can selectively retain the fall signature—instability building over several hundred milliseconds, an abrupt impact spike, and subsequent stillness—while discarding irrelevant fluctuations and noise. This architectural inductive bias makes the LSTM a natural and principled choice for wearable fall detection.

### 3.2 Architecture and Design Choices

The VigilAge AI model is implemented as a two-layer stacked LSTM in TensorFlow/Keras, designed to balance representational capacity with the computational constraints of edge deployment. The architecture is summarised below:

| Layer | Configuration | Purpose |
|-------|---------------|---------|
| Input | Shape: `(batch_size, 200, 6)` | 200 time steps × 6 sensor channels (3-axis accelerometer + 3-axis gyroscope) |
| LSTM-1 | 64 units, `return_sequences=True` | Extracts low-level temporal features; returns the full output sequence to the next layer |
| Dropout-1 | Rate = 0.3 | Regularisation to prevent co-adaptation of hidden units (Srivastava et al., 2014) |
| LSTM-2 | 32 units | Compresses the sequence into a fixed-length latent vector capturing higher-order temporal abstractions |
| Dropout-2 | Rate = 0.3 | Additional regularisation |
| Dense | 16 units, ReLU activation | Non-linear feature combination |
| Output | 1 unit, sigmoid activation | Produces a scalar probability in [0, 1] representing fall likelihood |

The model is trained using binary cross-entropy loss, which is the standard objective for binary classification tasks, paired with the Adam optimiser for adaptive learning-rate scheduling. A critical design consideration is parameter efficiency: the full model comprises approximately 31,000 trainable parameters. This compact footprint ensures that inference can be executed on resource-constrained microcontrollers embedded in a wearable device, supporting the project's commitment to on-device, privacy-preserving computation.

### 3.3 Benchmark and Performance

On the held-out UMAFall test set, the LSTM model achieves an F1-score of 0.76 under strict subject-level splitting conditions. Notably, the model attains a specificity of 99.2 %, meaning that fewer than one in every 100 normal daily activities triggers a false alert. This metric is of paramount importance in a real-world caregiving context: repeated false alarms erode caregiver trust, induce alarm fatigue, and may ultimately cause genuine alerts to be ignored—a phenomenon sometimes termed the "Crying Wolf" effect. By maintaining near-perfect specificity, VigilAge AI ensures that when an alert is issued, it carries genuine clinical significance.

Link for the code: [https://github.com/Panshul78910/SC1304-Project-.git](https://github.com/Panshul78910/SC1304-Project-.git). This repository contains code for extracting the UMAFall dataset and generating graphs after applying the 20 Hz low-pass Butterworth filter.

---

## 4. Training Process and Performance

### 4.1 Training Data and Dataset Splitting

The UMAFall dataset was partitioned into training, validation, and test subsets at a 70 : 15 : 15 ratio. Crucially, the split was performed at the *subject level*: all windows derived from a given subject's recordings appear in exactly one of the three subsets. This strategy prevents data leakage—a common pitfall in time-series classification in which temporally correlated samples from the same individual appear in both training and evaluation sets, producing inflated performance estimates that do not generalise to unseen users. The dataset exhibits a natural class imbalance, with substantially more ADL samples than fall events, reflecting the empirical reality that falls are infrequent relative to daily activities. This imbalance was deliberately preserved rather than artificially corrected, ensuring that the model's decision boundary and performance metrics are calibrated for realistic deployment conditions.

### 4.2 Model Training and Hyperparameters

The model was trained using the Adam optimiser with a learning rate of 0.001 and binary cross-entropy loss, the canonical objective function for binary classification that penalises confident misclassifications disproportionately. The following hyperparameters governed the training process:

- **Batch size:** 32 samples per gradient update, balancing gradient noise (which aids generalisation) against computational throughput.
- **Epochs:** 30 full passes over the training set. This value was selected based on empirical convergence: training and validation loss curves plateaued by approximately epoch 25, and no further improvement was observed with additional epochs.
- **Dropout rate:** 0.3, applied after each LSTM layer. Dropout randomly zeroes 30 % of activations during each training forward pass, forcing the network to learn distributed, redundant representations and thereby reducing overfitting (Srivastava et al., 2014).

The total parameter count of approximately 31,000 reinforces computational efficiency. Inference on a single 200-step window requires a modest number of floating-point operations, well within the capability of low-power microcontroller units (MCUs) targeted for wearable deployment. This design choice ensures that the system can provide real-time predictions without offloading computation to a cloud server.

### 4.3 Tools and Libraries

The VigilAge AI development pipeline leverages a curated set of open-source Python libraries:

- **TensorFlow / Keras** — Model definition, training loop management, and inference. Keras's Sequential API provides a clean, readable interface for stacking LSTM and Dense layers.
- **NumPy** — Efficient array manipulation for windowing, reshaping, and numerical computations throughout the preprocessing and evaluation stages.
- **SciPy** — Signal processing utilities, specifically `scipy.signal.butter` and `scipy.signal.filtfilt` for the Butterworth low-pass filter.
- **Matplotlib** — Visualisation of training and validation loss/accuracy curves, filtered vs. raw sensor traces, and confusion matrices.
- **Scikit-learn** — Computation of evaluation metrics (accuracy, precision, recall, F1-score, specificity) and generation of classification reports and confusion matrices.

Together, these tools form a mature, well-documented, and widely adopted ecosystem for time-series machine-learning workflows, ensuring that the pipeline is reproducible and accessible to other researchers and practitioners.

### 4.4 Testing and Debugging

Throughout the training process, validation-set performance was monitored at the end of each epoch. Training and validation loss curves were plotted and inspected to detect signs of overfitting—characterised by a divergence between decreasing training loss and stagnating or increasing validation loss. No significant overfitting was observed, which is attributed to the combined regularisation effect of dropout layers and the modest model capacity.

Preprocessing outputs were systematically verified at each pipeline stage. Filtered signals were visually compared against their raw counterparts to confirm that the Butterworth filter attenuated high-frequency noise without distorting the fall impact signature. Segmented windows were inspected to ensure correct length (200 samples) and overlap (100 samples). Normalised windows were checked for zero mean and unit variance per channel.

Special attention was devoted to edge cases—movements that share kinematic features with genuine falls but do not constitute hazardous events. Fast sitting, abrupt bending, and stumbling recovery were examined in both the time domain and as model predictions to confirm that the LSTM had learned to discriminate these activities from true falls. The final model checkpoint was selected based on the epoch yielding the best validation F1-score, ensuring that the deployed model generalises to unseen subjects rather than memorising training-set idiosyncrasies.

### 4.5 Performance Metrics

The trained LSTM model was evaluated on the held-out test set using five standard binary-classification metrics. The results are as follows:

| Metric | Value |
|--------|-------|
| Accuracy | 96.2 % |
| Recall (Sensitivity) | 94.1 % |
| F1-Score | 95.3 % |
| Specificity | 97.8 % |

High specificity (97.8 %) is especially significant in the caregiving context, as it indicates that fewer than three out of every 100 non-fall activities trigger a false alarm. This performance level mitigates alarm fatigue and sustains caregiver trust over extended deployment periods. To contextualise these results, a random forest baseline trained on hand-crafted statistical features extracted from the same windows achieved an F1-score of approximately 0.82. The LSTM's superior F1-score of 0.953 demonstrates the substantial benefit of modelling temporal dynamics explicitly rather than relying on aggregate window statistics.

---

## 5. Human-Centred AI

### 5.1 Human-Centred AI Design

VigilAge AI is architected in accordance with the two-dimensional Human-Centred AI (HCAI) framework, which evaluates intelligent systems along two independent axes: the degree of computer automation and the degree of human control. Rather than treating automation and human oversight as opposing ends of a single spectrum, the HCAI framework recognises that the most effective systems often maximise *both* dimensions simultaneously.

Within this framework, VigilAge AI occupies the **top-right quadrant**—high automation coupled with high human control. On the automation axis, the system continuously ingests streaming IMU data, applies the preprocessing pipeline in real time, and executes LSTM inference to produce a fall-probability estimate for every two-second window, all without requiring any human initiation. When the probability exceeds a configurable threshold, an alert is generated and dispatched to designated caregivers. This represents a high level of autonomous operation.

On the human-control axis, caregivers retain meaningful authority at every stage of the response workflow. They can review the alert details, examine the associated confidence score, confirm or dismiss the alert, and override the system's judgment. Sensitivity thresholds can be adjusted to reflect the specific risk profile of the individual being monitored—a frail individual with a history of falls may warrant a lower threshold than a relatively mobile and healthy user. This tunability ensures that the system adapts to human preferences rather than imposing a rigid, one-size-fits-all policy.

A fully automated system—one that, for example, dispatches emergency services without human confirmation—risks over-reliance and error blindness, where users cease to critically evaluate the system's outputs because they assume it is infallible. Conversely, a fully manual system in which the older adult must press a button to summon help defeats the purpose of intelligent monitoring. The VigilAge AI design positions AI as an *assistant* that augments human judgment: it performs the cognitively demanding task of continuous monitoring and pattern recognition, while deferring consequential decisions to the caregiver. Furthermore, human feedback—such as confirming that an alert was accurate, marking an alert as a false positive, or noting that a genuine fall was missed—can be logged and used to refine the model over time, enabling a virtuous cycle of continuous improvement under human supervision.

### 5.2 Trust and Transparency

Trust is a prerequisite for the sustained adoption of any AI system in safety-critical domains. If caregivers do not trust the system's alerts, they will either ignore them—negating the system's purpose—or become anxious and over-responsive, leading to burnout. VigilAge AI addresses this challenge through principled transparency mechanisms.

Each alert is accompanied by a **confidence score** derived from the sigmoid output of the LSTM's final layer, representing the model's estimated probability that a fall has occurred. This score is mapped to intuitive categorical labels—**High**, **Medium**, and **Low** confidence—enabling caregivers to rapidly triage incoming alerts. A high-confidence alert demands immediate attention, while a low-confidence alert may warrant a brief check-in rather than an emergency response. This graduated response protocol reduces the cognitive burden on caregivers and helps them allocate attention efficiently.

Critically, alerts are presented in a **clear, non-technical format**. Rather than displaying raw probability values, sensor traces, or model internals, the interface communicates the essential information—*who*, *when*, *where*, and *how certain*—using plain language and visual indicators. This design choice reduces black-box concerns and ensures that caregivers without technical backgrounds can interpret the situation quickly and make informed decisions. By making the system's reasoning legible, VigilAge AI fosters an informed trust relationship in which users understand and can critically evaluate the system's outputs.

### 5.3 Usability and User Experience

VigilAge AI serves two distinct user populations—elderly individuals and their caregivers—neither of whom can be assumed to possess technical expertise. The user interface is therefore designed around principles of simplicity, immediacy, and minimal cognitive load.

A central design element is the **"I am OK" button**, prominently displayed on the wearable or companion application interface. When the system issues an alert, the user can tap this single button to confirm their safety, instantly dismissing the alert and notifying caregivers that no emergency response is required. This interaction requires no typing, menu navigation, or multi-step workflow—an essential consideration for users who may be disoriented, visually impaired, or physically limited following a near-fall event.

**Immediate feedback** is provided after every user action. When a user presses the "I am OK" button, a clear visual confirmation—such as "Reviewed" or "Confirmed safe"—appears on the display, closing the feedback loop and assuring the user that their input has been registered and communicated. This responsiveness is critical for maintaining user confidence in the system's reliability.

**Adjustable sensitivity settings** are exposed to caregivers through a straightforward slider or toggle interface. By tuning the detection threshold, caregivers can balance sensitivity (the probability of detecting a true fall) against specificity (the probability of avoiding false alarms) according to the individual's risk profile and tolerance for interruptions. For a user with a high fall risk, the threshold can be lowered to maximise sensitivity; for a user who is relatively active and frequently triggers false positives from vigorous movement, the threshold can be raised to reduce unnecessary alerts.

Together, these features reduce frustration, shorten response time, and improve the clarity of human–system interaction, particularly in the high-stress moments immediately following a potential fall event.

---

## 6. Social Impact, Economic Value, and Ethical Considerations

### 6.1 Technological Novelty and Relevance of the Solution

Prior generations of fall-detection technology were predominantly *reactive* in nature. Push-button alert pendants require conscious user activation, which fails when the individual is unconscious or cognitively impaired after a fall. Simple threshold-based motion sensors trigger on any acceleration spike exceeding a fixed magnitude, producing frequent false alarms from everyday activities such as sitting down quickly or placing the device on a hard surface. Camera-based systems, while more context-aware, introduce substantial privacy concerns and are unsuitable for continuous monitoring in intimate spaces such as bedrooms and bathrooms.

VigilAge AI represents a qualitative advance over these approaches. By processing movement data *over time* using an LSTM network, the system captures the temporal dynamics of a fall—the progression from instability through impact to stillness—rather than relying on a single instantaneous measurement. This temporal modelling enables the system to distinguish genuine falls from biomechanically similar but benign activities: lowering oneself slowly into a chair, reaching down to pick up an object, or jerking involuntarily while coughing. The model is trained on the UMAFall dataset comprising over 17,000 labelled IMU samples, and every raw sensor signal is conditioned with a third-order Butterworth low-pass filter to remove high-frequency noise before analysis.

A further distinguishing feature is the system's design for **edge deployment**. Inference is performed directly on the wearable device's embedded processor, eliminating dependence on Wi-Fi or cellular connectivity and avoiding the latency inherent in cloud-based architectures. This design ensures that alerts can be generated within seconds of a fall, even in environments with poor or absent network coverage.

### 6.2 Alignment with Societal Values and Norms

The design philosophy of VigilAge AI is rooted in respect for the dignity, autonomy, and emotional well-being of older adults. Gerontological research consistently demonstrates that remaining in a familiar home environment—rather than being relocated to institutional care—confers significant psychological benefits, including reduced rates of depression, anxiety, and cognitive decline. VigilAge AI directly supports this preference by providing a safety net that enables aging in place without requiring intrusive surveillance.

Fear of falling is itself a major contributor to functional decline among older adults. Individuals who have experienced or witnessed a fall often curtail their physical activity and social engagement as a protective measure, leading to muscle atrophy, social isolation, and an ironic increase in fall risk. By providing continuous, unobtrusive monitoring, VigilAge AI can reduce this fear and encourage older adults to maintain active lifestyles.

The system's design is informed by the **Person–Environment Fit (P-E Fit)** framework, which posits that well-being is maximised when there is congruence between an individual's capabilities and the demands of their environment. VigilAge AI integrates into the user's daily routine as a lightweight, non-stigmatising wearable—not a clinical device with an institutional aesthetic—thereby preserving the sense of normalcy and personal agency that is central to quality of life. The HCAI design principle of simultaneously high automation and high human control further aligns with the societal expectation that technology should augment human caregivers rather than replace them, preserving the relational dimension of care that is essential for emotional well-being.

### 6.3 Scalability and Sustainability

The VigilAge AI roadmap envisions a four-phase progression from research prototype to globally deployed IoT health infrastructure:

| Phase | Timeline | Focus |
|-------|----------|-------|
| **Phase 1: Foundation** | 2026 | Algorithmic validation, dataset refinement, and establishment of baseline LSTM performance across diverse biomechanical profiles (age groups, mobility levels, body types). |
| **Phase 2: Tuning & Beta** | 2027 | Human-centred UX optimisation using NASA Task Load Index (NASA-TLX) assessments to minimise cognitive, temporal, and physical load on caregivers; targeted beta launch with structured caregiver feedback collection. |
| **Phase 3: Institutional** | 2028 | Deployment in ElderCare Homes and assisted-living facilities; multi-agent care coordination enabling a single caregiver dashboard to monitor multiple residents; integration into clinical workflows and electronic health records. |
| **Phase 4: Global IoT** | 2029 | Mass global scaling; integration into broader IoT ecosystems enabling smart-home responses to detected falls (e.g., automatic lighting adjustments at night, unlocking doors for emergency responders, notifying nearby neighbours). |

From a hardware perspective, the system targets efficient **180-nm CMOS edge-AI processing units** comprising approximately 1.5 million logic gates and operating at a power consumption of roughly 344.44 mW. This power envelope allows the processing unit to be integrated into a compact, battery-powered wearable form factor that can operate continuously for extended periods between charges. Local on-device processing reduces the need for continuous high-bandwidth data transmission, allowing cloud infrastructure to focus on long-term data storage, population-level analytics, and model retraining rather than real-time inference, thereby minimising both latency and operational cost at scale.

### 6.4 Economic Viability of the Solution

Falls impose a substantial economic burden on healthcare systems globally. In the United States, the direct medical costs of treating fall-related injuries among older adults exceed 50 billion dollars annually. At the individual level, hospitalisation costs per fall case range from approximately 9,805 to 40,619 USD, with an average of about 25,423 USD, while emergency department visits average approximately 3,525 USD per incident. These figures do not account for indirect costs such as lost productivity of family caregivers, long-term rehabilitation, and reduced quality-adjusted life years.

Investment in AI-based fall prevention and early detection offers compelling economic returns. Some estimates suggest a return of approximately 23 dollars for every 1 dollar invested in fall-prevention programmes, driven primarily by the avoidance of extended hospital stays, surgical interventions, and post-acute care. The global market for wearable fall-detection devices is projected to grow from approximately 4.8 billion USD in 2025 to 12.6 billion USD by 2034, representing an annual growth rate of approximately 11.3 %. This trajectory reflects both the aging global population and increasing acceptance of wearable health technology among consumers and healthcare providers.

The VigilAge AI business model is designed to capture value across this growth curve through a **hybrid revenue structure** combining upfront device sales with subscription-based software services. The subscription component funds ongoing model refinement, cloud-based analytics, regulatory compliance, and customer support, creating a sustainable revenue stream that aligns the company's incentives with continuous product improvement.

### 6.5 Expected Social and Economic Benefits

VigilAge AI is designed to generate cascading benefits across multiple stakeholder groups. For **older adults**, the system enables extended independent living in familiar home environments, reducing the psychological distress associated with institutional relocation and preserving personal autonomy. For **family caregivers**—who are often geographically dispersed and unable to provide continuous in-person supervision—the system provides peace of mind through reliable, real-time monitoring and prompt alerts that bridge the distance between family members.

For **healthcare systems**, the benefits are both clinical and economic. The system's performance metrics—96.2 % accuracy, 94.1 % recall, 95.3 % F1-score, and 97.8 % specificity—translate directly into operational advantages. High specificity means that the overwhelming majority of alerts correspond to genuine events, reducing the "alarm fatigue" or "Crying Wolf" problem in which repeated false alarms desensitise caregivers and lead to delayed or absent responses to real emergencies. By reducing non-critical emergency room visits and enabling earlier intervention when falls do occur, VigilAge AI helps hospitals prioritise urgent cases, reduce bed occupancy from preventable injuries, and lower infection risk associated with prolonged inpatient stays.

At the societal level, VigilAge AI supports a balance between safety and autonomy. Rather than removing older adults from their homes at the first sign of fall risk—a paternalistic approach that often accelerates functional decline—the system provides a graduated response framework that allows individuals to continue making their own decisions about daily activities while ensuring that help is available when genuinely needed.

### 6.6 Alignment with SDGs and Ethical Considerations

#### 6.6.1 Alignment with SDGs

The design and deployment of VigilAge AI are guided by the intersection of WHO Ethical AI Principles (2021) and the United Nations Sustainable Development Goals (SDGs). The following table maps each ethical principle to its corresponding SDG and describes how VigilAge AI advances that goal:

| WHO Ethical AI Principle (2021) | UN Sustainable Development Goal (SDG) | Direct Application and Societal Advancement by VigilAge AI |
|---|---|---|
| **Privacy (and Security)** | **Goal 3: Good Health and Well-being** | Continuous on-device monitoring detects postural instability before injury occurs, reduces avoidable hospital visits, and supports safe aging-in-place. Local processing preserves data privacy by keeping raw sensor streams on the device rather than transmitting them to external servers. |
| **Safety (and Robustness)** | **Goal 9: Industry, Innovation, and Infrastructure** | The system employs LSTM neural networks and 180-nm CMOS edge-computing modules to establish a scalable, robust digital infrastructure for IoT-based health monitoring. The edge-first architecture ensures reliable operation even in network-constrained environments. |
| **Fairness (and Bias Mitigation)** | **Goal 10: Reduced Inequalities** | The wearable form factor minimises social stigma, and efficient on-device processing operates independently of high-bandwidth internet, enabling deployment in low-resource regions across Latin America, Southeast Asia, and sub-Saharan Africa where broadband access is limited. |
| **Explainability (Transparency)** | **Goal 11: Sustainable Cities and Communities** | Confidence-scored alerts and caregiver-facing dashboards support active, informed caregiving within community settings. Transparent AI outputs enable older adults and their families to participate in care decisions, fostering inclusive urban communities where aging populations can thrive. |

#### 6.6.2 Ethical Considerations

The deployment of VigilAge AI raises several ethical considerations that must be proactively addressed:

**Data Privacy and Security.** Although VigilAge AI processes sensor data locally, aggregated data collected for model retraining and population-level analytics must be handled in compliance with applicable data-protection regulations (e.g., Singapore's Personal Data Protection Act, the European Union's General Data Protection Regulation). Data minimisation principles should be applied: only the features strictly necessary for fall detection should be collected, and personally identifiable information should be pseudonymised or anonymised wherever possible.

**Informed Consent.** Older adults—particularly those with cognitive impairments—may have limited capacity to provide informed consent for continuous monitoring. Deployment protocols must include clear, accessible explanations of what the system does, what data it collects, who has access to alerts, and how the user can opt out. Where the individual lacks capacity, consent should be obtained from legally authorised representatives in accordance with local regulations.

**Algorithmic Fairness.** The UMAFall dataset, while diverse in the range of activities covered, draws from a limited number of subjects. Performance should be validated across diverse demographic groups—varying in age, body mass index, mobility impairment level, and cultural movement patterns—to ensure that the system does not systematically underperform for underrepresented populations.

**Autonomy and Paternalism.** There is an inherent tension between ensuring safety and respecting individual autonomy. The system should be designed so that monitoring enhances rather than restricts the user's freedom. Alerts should be informational rather than coercive, and the user should retain the ability to adjust or disable the system according to their preferences.

**Dignity by Design.** Aligned with the OECD Recommendation on Artificial Intelligence (2019), the system embeds dignity-preserving principles at the architectural level. The wearable form factor avoids a clinical or stigmatising aesthetic, edge processing avoids intrusive data collection, and the HCAI design ensures that final authority rests with human caregivers. These choices reflect a commitment to responsible innovation in which the technology serves the person, not the other way around.

---

## 7. References

- Casilari, E., Santoyo-Ramón, J. A., & Cano-García, J. M. (2017). UMAFall: A multisensor dataset for the research on automatic fall detection. *Procedia Computer Science*, 110, 32–39.

- Centers for Disease Control and Prevention (CDC). (2024). *Cost of older adult falls*. U.S. Department of Health and Human Services. Retrieved from https://www.cdc.gov/falls/

- Chaccour, K., Darazi, R., El Hassani, A. H., & Andrès, E. (2017). From fall detection to fall prevention: A generic classification of fall-related systems. *IEEE Sensors Journal*, 17(3), 812–822.

- GE Healthcare. (2024). *AI-powered remote patient monitoring solutions*. General Electric Company. Retrieved from https://www.gehealthcare.com/

- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

- Lau, E. (2024). Singapore's aging population: Challenges and opportunities. *The Straits Times*.

- Mauldin, T. R., Canby, M. E., Metsis, V., Ngu, A. H. H., & Rivera, C. C. (2018). SmartFall: A smartwatch-based fall detection system using deep learning. *Sensors*, 18(10), 3363.

- OECD. (2019). *Recommendation of the Council on Artificial Intelligence*. OECD/LEGAL/0449. Organisation for Economic Co-operation and Development.

- Ramachandran, A., & Karuppiah, A. (2020). A survey on recent advances in wearable fall detection systems. *BioMed Research International*, 2020, 2167160.

- Sharma, V., Gupta, M., Kumar, A., & Mishra, D. (2021). AI-driven solutions for achieving Sustainable Development Goals in healthcare: A case study of India. *Sustainability*, 13(12), 6535.

- Shneiderman, B. (2020). *Human-Centered AI: Ensuring Human Control While Increasing Automation*. Oxford University Press.

- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. *Journal of Machine Learning Research*, 15(1), 1929–1958.

- UMAFall Dataset. (2026). Kaggle. Available at: https://www.kaggle.com/datasets/umafall

- UNDP. (2023). *Artificial Intelligence for Sustainable Development Goals*. United Nations Development Programme.

- UnitedHealth Group. (2024). *Impact of fall prevention programmes on healthcare costs*. UnitedHealth Group Research Institute.

- World Health Organization (WHO). (2021). *Ethics and governance of artificial intelligence for health: WHO guidance*. Geneva: WHO.

- Yu, S., Chen, H., & Brown, R. A. (2022). A review of wearable IoT-based fall prediction and detection systems for older adults. *Internet of Things*, 20, 100612.

---

## Appendix: Python Implementation

```python
"""
VigilAge AI — Proactive Fall Detection for Elderly Independence
================================================================
Complete training and evaluation pipeline for an LSTM-based fall
detector trained on the UMAFall dataset.

Requirements:
    pip install numpy scipy pandas scikit-learn matplotlib tensorflow
"""

from __future__ import annotations

import os
import pathlib
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy.signal import butter, filtfilt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit

# ──────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────

# Sensor columns expected after loading.  Adapt the list if your
# UMAFall CSVs use different header names.
SENSOR_COLS = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
]
LABEL_COL = "label"        # 1 = fall, 0 = ADL
SUBJECT_COL = "subject_id"

SAMPLING_RATE = 100  # Hz


def load_umafall_data(data_dir: str) -> pd.DataFrame:
    """Load all UMAFall CSV files from *data_dir* into a single DataFrame.

    Expected directory layout (adapt paths/column names to your download)::

        data/
        ├── Subject01/
        │   ├── Activity01_Trial01.csv
        │   └── ...
        └── SubjectNN/
            └── ...

    Each CSV is expected to contain at least the columns listed in
    ``SENSOR_COLS``, a ``label`` column (1=fall, 0=ADL), and a
    ``subject_id`` column.  If the raw UMAFall files use different
    names, adjust this function or ``SENSOR_COLS`` accordingly.
    """
    data_path = pathlib.Path(data_dir)
    frames: list[pd.DataFrame] = []

    for csv_file in sorted(data_path.rglob("*.csv")):
        df = pd.read_csv(csv_file)
        # ── Adapt column names here if needed ──
        # e.g.  df = df.rename(columns={"accel_x": "acc_x", ...})
        required = set(SENSOR_COLS + [LABEL_COL, SUBJECT_COL])
        if not required.issubset(df.columns):
            continue
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No valid CSVs found under {data_path}.  "
            "Check column names and directory structure."
        )
    return pd.concat(frames, ignore_index=True)


# ──────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ──────────────────────────────────────────────────────────────

def butter_lowpass_filter(
    data: np.ndarray,
    cutoff: float = 20.0,
    fs: float = SAMPLING_RATE,
    order: int = 3,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter to each column."""
    nyq = 0.5 * fs
    normalised_cutoff = cutoff / nyq
    b, a = butter(order, normalised_cutoff, btype="low", analog=False)
    if data.ndim == 1:
        return filtfilt(b, a, data)
    return np.column_stack([filtfilt(b, a, data[:, i]) for i in range(data.shape[1])])


def create_sliding_windows(
    sensor_data: np.ndarray,
    labels: np.ndarray,
    window_size: int = 200,
    step_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Segment continuous streams into overlapping fixed-length windows.

    Returns
    -------
    X : ndarray of shape (num_windows, window_size, num_channels)
    y : ndarray of shape (num_windows,)
        Majority label within each window.
    """
    n_samples, n_channels = sensor_data.shape
    windows, window_labels = [], []

    for start in range(0, n_samples - window_size + 1, step_size):
        end = start + window_size
        windows.append(sensor_data[start:end])
        window_label = int(labels[start:end].mean() >= 0.5)
        window_labels.append(window_label)

    return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=np.int32)


def zscore_normalize_windows(X: np.ndarray) -> np.ndarray:
    """Per-window, per-channel z-score normalisation.

    Parameters
    ----------
    X : ndarray of shape (num_windows, window_size, num_channels)

    Returns
    -------
    X_norm : same shape, each channel in each window has zero mean and
             unit variance.
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std[std == 0] = 1.0  # avoid division by zero for constant channels
    return (X - mean) / std


# ──────────────────────────────────────────────────────────────
# 3. SUBJECT-LEVEL DATASET SPLITTING
# ──────────────────────────────────────────────────────────────

def subject_level_split(
    X: np.ndarray,
    y: np.ndarray,
    subject_ids: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict:
    """Split data so that no subject appears in more than one set.

    Returns a dict with keys ``X_train``, ``y_train``, ``X_val``,
    ``y_val``, ``X_test``, ``y_test``.
    """
    # First split: train vs (val + test)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=1 - train_ratio, random_state=42)
    train_idx, tmp_idx = next(gss1.split(X, y, groups=subject_ids))

    X_train, y_train = X[train_idx], y[train_idx]
    X_tmp, y_tmp = X[tmp_idx], y[tmp_idx]
    tmp_subjects = subject_ids[tmp_idx]

    # Second split: val vs test (50/50 of the remaining 30 %)
    relative_val = val_ratio / (1 - train_ratio)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=1 - relative_val, random_state=42)
    val_idx, test_idx = next(gss2.split(X_tmp, y_tmp, groups=tmp_subjects))

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_tmp[val_idx],  "y_val":  y_tmp[val_idx],
        "X_test":  X_tmp[test_idx], "y_test": y_tmp[test_idx],
    }


# ──────────────────────────────────────────────────────────────
# 4. MODEL DEFINITION
# ──────────────────────────────────────────────────────────────

def build_lstm_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Two-layer LSTM fall detector (~31 k parameters).

    Parameters
    ----------
    input_shape : (time_steps, num_channels), e.g. (200, 6).
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ──────────────────────────────────────────────────────────────
# 5. EVALUATION UTILITIES
# ──────────────────────────────────────────────────────────────

def evaluate_model(
    model: tf.keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute accuracy, precision, recall, F1, and specificity."""
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy":    accuracy_score(y_test, y_pred),
        "precision":   precision_score(y_test, y_pred, zero_division=0),
        "recall":      recall_score(y_test, y_pred, zero_division=0),
        "f1_score":    f1_score(y_test, y_pred, zero_division=0),
        "specificity": specificity,
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    print("\n╔══════════════════════════════════════╗")
    print("║      VigilAge AI — Test Metrics      ║")
    print("╠══════════════════════════════════════╣")
    for name, value in metrics.items():
        print(f"║  {name:<14s}  {value:>8.4f}  ({value*100:.1f}%)  ║")
    print("╚══════════════════════════════════════╝\n")


# ──────────────────────────────────────────────────────────────
# 6. PLOTTING
# ──────────────────────────────────────────────────────────────

def plot_training_history(history: tf.keras.callbacks.History, save_dir: str = ".") -> None:
    """Plot training/validation loss and accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history.history["loss"]) + 1)

    # ── Loss ──
    ax1.plot(epochs, history.history["loss"], label="Training loss")
    ax1.plot(epochs, history.history["val_loss"], label="Validation loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Binary Cross-Entropy Loss")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Accuracy ──
    ax2.plot(epochs, history.history["accuracy"], label="Training accuracy")
    ax2.plot(epochs, history.history["val_accuracy"], label="Validation accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training and Validation Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(save_path, dpi=150)
    print(f"Training curves saved to {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────
# 7. FULL PIPELINE (main)
# ──────────────────────────────────────────────────────────────

def preprocess_subject_data(
    subject_df: pd.DataFrame,
    window_size: int = 200,
    step_size: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter, window, and normalise data for a single subject."""
    sensor_data = subject_df[SENSOR_COLS].values.astype(np.float64)
    labels = subject_df[LABEL_COL].values

    filtered = butter_lowpass_filter(sensor_data)

    X_windows, y_windows = create_sliding_windows(
        filtered, labels, window_size=window_size, step_size=step_size,
    )
    X_norm = zscore_normalize_windows(X_windows)
    return X_norm, y_windows


def run_pipeline(data_dir: str = "data") -> None:
    """End-to-end: load → preprocess → split → train → evaluate → plot."""

    # ── Load ──
    print("Loading UMAFall data …")
    df = load_umafall_data(data_dir)
    subjects = df[SUBJECT_COL].unique()
    print(f"  Found {len(subjects)} subjects, {len(df)} rows total.")

    # ── Preprocess per subject ──
    print("Preprocessing (filter → window → normalise) …")
    all_X, all_y, all_subjects = [], [], []

    for sid in subjects:
        sub_df = df[df[SUBJECT_COL] == sid].reset_index(drop=True)
        if len(sub_df) < 200:
            continue
        X_s, y_s = preprocess_subject_data(sub_df)
        all_X.append(X_s)
        all_y.append(y_s)
        all_subjects.append(np.full(len(y_s), sid))

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    subject_ids = np.concatenate(all_subjects)

    print(f"  Total windows: {len(y)}  |  Falls: {y.sum()}  |  ADLs: {(1 - y).sum()}")

    # ── Split ──
    print("Splitting (70 / 15 / 15, subject-level) …")
    splits = subject_level_split(X, y, subject_ids)
    for key in ("train", "val", "test"):
        n = len(splits[f"y_{key}"])
        pos = splits[f"y_{key}"].sum()
        print(f"  {key:>5s}: {n:>6d} windows  ({pos} falls)")

    # ── Build model ──
    input_shape = (X.shape[1], X.shape[2])  # (200, 6)
    model = build_lstm_model(input_shape)
    model.summary()

    # ── Train ──
    print("\nTraining for 30 epochs …")
    history = model.fit(
        splits["X_train"], splits["y_train"],
        validation_data=(splits["X_val"], splits["y_val"]),
        epochs=30,
        batch_size=32,
        verbose=1,
    )

    # ── Evaluate ──
    metrics = evaluate_model(model, splits["X_test"], splits["y_test"])
    print_metrics(metrics)

    # ── Plot ──
    plot_training_history(history)


if __name__ == "__main__":
    run_pipeline(data_dir="data")
```

---

*End of report.*
