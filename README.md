# Awesome Vision Language Action

🔥 Latest Advances on Vision-Language-Action Models.  
Embodied intelligence is one of the most critical carriers for general artificial intelligence to interact with the physical world. This repository curates a comprehensive list of research papers, models, datasets, and resources related to Vision-Language-Action (VLA) models, with a focus on robotics and embodied AI.  

---

## Table of Contents
- [Awesome Vision Language Action](#awesome-vision-language-action)
  - [Table of Contents](#table-of-contents)
  - [Milestone Papers](#milestone-papers)
  - [VLA Model Architectures](#vla-model-architectures)
    - [General VLAs](#general-vlas)
    - [Humanoid Robot VLAs](#humanoid-robot-vlas)
    - [Robotic Foundation Models](#robotic-foundation-models)
  - [Components of VLA Systems](#components-of-vla-systems)
    - [Reinforcement Learning](#reinforcement-learning)
    - [Pretrained Visual Representations](#pretrained-visual-representations)
    - [World Models](#world-models)
  - [Datasets \& Benchmarks](#datasets--benchmarks)
  - [Tutorials \& Courses](#tutorials--courses)
  - [Related Surveys](#related-surveys)
  - [Contributing](#contributing)

---

## Milestone Papers
| Date       | Keywords          | Institute   |   Paper     |Code      |
| :-----: | :-------------------: | :-------------------: | ---------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------: |
| 2022-12    | Transformer                | Google               | [RT-1: Robotics Transformer for Real-World Control at Scale](https://arxiv.org/abs/2212.06817)                                     | [Code](https://github.com/google-research/robotics_transformer)                               |
| 2023-03    | Multimodal Embodied LLM    | Google               | [PaLM-E: An Embodied Multimodal Language Model](https://arxiv.org/abs/2303.03378)                                                  | -                                                                                             |
| 2023-04    | Diffusion Policy           | Columbia University   | [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137)                              | [Code](https://github.com/real-stanford/diffusion_policy)                                    |
| 2023-04 |    Action Chunking    |       Stanford        | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT](https://arxiv.org/pdf/2304.13705)                        |                       [tonyzhaozh/act](https://github.com/tonyzhaozh/act)                       |
| 2023-07    | Web Knowledge Transfer     | Google DeepMind      | [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818)                  | -                                                                                             |
| 2023-10 |  Action-in-Video-Out  |      UC Berkeley      | [UniSim: Learning Interactive Real-World Simulators](https://arxiv.org/pdf/2310.06114)                                             |                                                -                                                |
| 2024-03 |  Vector Quantization  |  New York University  | [Behavior Generation with Latent Actions](https://arxiv.org/pdf/2403.03181)                                                        |           [jayLEE0301/vq_bet_official](https://github.com/jayLEE0301/vq_bet_official)           |
| 2024-05 |    Low-Cost/Design    |        Google         | [ALOHA 2: An Enhanced Low-Cost Hardware for Bimanual Teleoperation](https://arxiv.org/abs/2405.02292)                              |            [tonyzhaozh/aloha](https://github.com/tonyzhaozh/aloha/tree/main/aloha2)             |
| 2024-06    | Open-Source VLA           | Stanford & UC Berkeley | [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)                                           | [Code](https://github.com/openvla/openvla)                                                   |
| 2024-09 |     Heterogeneous     |          MIT          | [Heterogenous Pre-trained Transformers](https://arxiv.org/pdf/2409.20537)                                                          |                           [liruiw/HPT](https://github.com/liruiw/HPT)                           |
| 2024-10    | Flow Matching             | Physical Intelligence | [π₀: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164)                              | [Code](https://github.com/Physical-Intelligence/openpi)                                      |
| 2024-10 |  Bimanual Diffusion   |          THU          | [RDT-1B: a Diffusion Foundation Model for Bimanual Manipulation](https://arxiv.org/pdf/2410.07864)                                 |  [thu-ml/RoboticsDiffusionTransformer](https://github.com/thu-ml/RoboticsDiffusionTransformer)  |
| 2024-10 | Video-Language-Action |       ByteDance       | [GR-2: A Generative Video-Language-Action Model with Web-Scale Knowledge for Robot Manipulation](https://arxiv.org/abs/2410.06158) |                                                -                                                |
| 2023-12  | Language Agents, Reinforcement Learning    | -         | [Paper](http://arxiv.org/abs/2303.11366v4)    | [Code](https://github.com/noahshinn/reflexion) |
| 2023-12  | Visual Cortex, Embodied Intelligence       | -         | [Paper](http://arxiv.org/abs/2303.18240v2)    | [Code](https://github.com/facebookresearch/eai-vc) |
| 2023-10  | Robot Learning, Human Feedback             | -         | [Paper](http://arxiv.org/abs/2307.15801v2)    | [Code](https://github.com/mj-hwang/seed) |
| 2023-07  | Language-Driven Learning, Robotics          | -         | [Paper](https://arxiv.org/abs/2302.12766)     | -    |
| 2022-11  | Masked Pre-training, Robot Learning        | -         | [Paper](http://arxiv.org/abs/2203.06173v1)    | [Code](https://github.com/ir413/mvp) |
| 2022-11  | Visual Representation, Robot Manipulation  | -         | [Paper](http://arxiv.org/abs/2203.12601v3)    | [Code](https://github.com/facebookresearch/r3m) |
| 2022-07  | Pre-Trained Vision Models, Control         | -         | [Paper](http://arxiv.org/abs/2203.03580v2)    | -    |
| 2021-12  | Reinforcement Learning, Transformers       | -         | [Paper](https://arxiv.org/abs/2106.01345)     | [Code](https://github.com/kzl/decision-transformer) |
| 2021-12  | Offline RL, Sequence Modeling              | -         | [Paper](http://arxiv.org/abs/2106.02039v4)     | [Code](https://github.com/Farama-Foundation/D4RL) |
| 2021-07  | Vision-Language Models, Transfer Learning  | -         | [Paper](http://arxiv.org/abs/2103.00020v1)    | [Code](https://github.com/openai/CLIP) |
| 2025-05  | Unified Pretraining, Gaussian Splatting    | -         | [Paper](http://arxiv.org/abs/2502.17860v2)    | [Code](https://github.com/Li-Hao-yuan/UniGS) |
| 2024-11  | Vision Foundation Models, Distillation     | -         | [Paper](http://arxiv.org/abs/2407.20179v2)    | -    |
| 2024-06  | 3D Gaussians, Generative Dynamics          | -         | [Paper](https://arxiv.org/abs/2311.12198)     | [Code](https://github.com/XPandora/PhysGaussian) |
| 2023-12  | Self-Supervised Learning, Visual Features   | -         | [Paper](http://arxiv.org/abs/2304.07193v2)    | [Code](https://github.com/facebookresearch/dinov2) |
| 2023-11  | Feature Fields, Language-Guided Manipulation | -       | [Paper](http://arxiv.org/abs/2308.07931v2)    | [Code](https://github.com/f3rm/f3rm) |
| 2023-11  | Sensorimotor Pre-training                   | -         | [Paper](https://arxiv.org/abs/2306.10007)     | -    |
| 2023-11  | Auditory Self-Supervision, Robot Manipulation | -      | [Paper](http://arxiv.org/abs/2210.01116v1)    | [Code](https://audio-robot-learning.github.io) |
| 2023-06  | Self-Supervised Learning, Joint-Embedding   | -         | [Paper](http://arxiv.org/abs/2301.08243v3)    | -    |
| 2023-05  | Visual Reward, Representation Learning      | -         | [Paper](https://arxiv.org/abs/2210.00030)     | [Code](https://github.com/facebookresearch/vip) |
| 2022-12  | Masked Autoencoding, Decision Making        | -         | [Paper](http://arxiv.org/abs/2211.12740v2)   | [Code](https://github.com/FangchenLiu/MaskDP_public) |
| 2024-07  | Multimodal Prompts, Robot Manipulation     | -         | [Paper](https://arxiv.org/abs/2310.09676)     | -    |
| 2024-05  | Video Generative Pre-training, Robot Manipulation | - | [Paper](http://arxiv.org/abs/2312.13139v2)    | -    |
| 2023-10  | Perception-Action Transformers, Robotics    | -         | [Paper](https://arxiv.org/abs/2209.11133)     | -    |
| 2023-10  | Visual Pre-training, Robot Manipulation    | -         | [Paper](https://arxiv.org/abs/2308.03620)     | -    |
| 2023-05  | Self-supervised Multi-task Learning, Control Transformers | - | [Paper](http://arxiv.org/abs/2301.09816v1)   | -    |
| 2023-05  | Transformer World Models, Sample Efficiency | -         | [Paper](http://arxiv.org/abs/2303.07109v1)    | -    |
| 2023-05  | Transformer World Models, Long-horizon Tasks | -        | [Paper](http://arxiv.org/abs/2209.00588v2)    | [Code](https://github.com/eloialonso/iris) |
| 2023-01  | World Models, Cross-domain Learning        | -         | [Paper](http://arxiv.org/abs/2301.04104v2)    | -    |
| 2022-12  | Video Pre-training, Unlabeled Videos       | -         | [Paper](http://arxiv.org/abs/2206.11795v1)    | -    |
| 2022-11  | World Models, Physical Robot Learning      | -         | [Paper](http://arxiv.org/abs/2206.14176v1)   | -    |
| 2022-06  | Autonomous Machine Intelligence            | -         | [Paper](https://openreview.net/pdf?id=BZ5a1r-kVsf) | -    |
| 2021-05  | Discrete World Models, Atari               | -         | [Paper](http://arxiv.org/abs/2010.02193v4)    | -    |
| 2020-05  | Latent Imagination, Behavior Learning      | -         | [Paper](http://arxiv.org/abs/1912.01603v3)   | -    |
| 2024-07  | Generative Interactive Environments        | -         | [Paper](https://arxiv.org/abs/2402.15391)     | -    |
| 2024-07  | 3D Vision-Language-Action World Model      | -         | [Paper](https://arxiv.org/abs/2403.09631)     | [Code](https://github.com/UMass-Embodied-AGI/3D-VLA) |
| 2024-05  | Efficient Task Planning, LLMs              | -         | [Paper](https://arxiv.org/abs/2310.08582)     | -    |
| 2024-05  | Interactive Real-World Simulators           | -         | [Paper](https://arxiv.org/abs/2310.06114)    | [Code](https://universal-simulator.github.io/unisim/) |
| 2024-03  | Retrieval Augmented Reasoning              | -         | [Paper](http://arxiv.org/abs/2403.05313v1)   | -    |
| 2023-12  | Embodied Instruction Following             | -         | [Paper](http://arxiv.org/abs/2312.07062v2)  | -    |
| 2023-12  | LLM-based World Models, Task Planning      | -         | [Paper](http://arxiv.org/abs/2305.14909v2)   | [Code](https://github.com/GuanSuns/LLMs-World-Models-for-Planning) |
| 2023-12  | LLM-World Model Integration                | -         | [Paper](http://arxiv.org/abs/2305.10626v3)   | [Code](https://github.com/Sfedfcv/redesigned-pancake) |
| 2023-12  | Language-Guided World Modelling            | -         | [Paper](http://arxiv.org/abs/2301.12050v2)   | [Code](https://github.com/DeckardAgent/deckard) |
| 2023-12  | Text-Guided Video Generation               | -         | [Paper](http://arxiv.org/abs/2302.00111v3)   | -    |
| 2023-12  | LLM-enhanced Planning                      | -         | [Paper](http://arxiv.org/abs/2305.14078v2)    | -    |
| 2023-11  | World Model Reasoning with LLMs            | -         | [Paper](https://arxiv.org/abs/2305.14992)    | -    |
| 2023-10  | Visual Affordance Grounding                | -         | [Paper](http://arxiv.org/abs/2210.01911v3)    | -    |
| 2023-05  | Reasoning-Action Synergy                   | -         | [Paper](http://arxiv.org/abs/2210.03629v3)    | -    |
| 2022-04  | Language-Conditioned Imitation Learning     | -         | [Paper](http://arxiv.org/abs/2204.06252v2)    | [Code](https://github.com/lukashermann/hulc) |
| 2021-11  | CLIP-based Robotic Manipulation            | -         | [Paper](http://arxiv.org/abs/2109.12098v1)    | [Code](https://github.com/cliport/cliport) |
| 2021-11  | Zero-Shot Task Generalization              | -         | [Paper](https://arxiv.org/abs/2202.02005)     | [Code](https://github.com/google-research/tensor2robot/tree/master/research/bcz) |
| 2020-11  | Visual Rearrangement Networks              | -         | [Paper](http://arxiv.org/abs/2010.14406v3)   | -    |
| 2025-01  | Vision-Language Models, Robot Imitation     | -         | [Paper](https://arxiv.org/abs/2311.01378)     | [Code](https://github.com/RoboFlamingo/RoboFlamingo) |
| 2024-10  | Diffusion Model, Bimanual Manipulation      | -         | [Paper](http://arxiv.org/abs/2410.07864v2)    | [Code](https://github.com/thu-ml/RoboticsDiffusionTransformer) |
| 2024-09  | Relational Keypoints, Spatio-Temporal Reasoning | -    | [Paper](http://arxiv.org/abs/2409.01652v2)    | [Code](https://github.com/huangwl18/ReKep) |
| 2024-07  | Multimodal Diffusion, Versatile Behavior    | -         | [Paper](http://arxiv.org/abs/2407.05996v1)   | [Code](https://github.com/intuitive-robots/mdt_policy) |
| 2024-07  | Generalist Robot Policy, Open-Source        | -         | [Paper](https://arxiv.org/abs/2405.12213)     | [Code](https://github.com/octo-models/octo) |
| 2024-07  | 3D Diffusion Policy, Generalizable Learning | -         | [Paper](http://arxiv.org/abs/2403.03954v7)   | [Code](https://github.com/YanjieZe/3D-Diffusion-Policy) |
| 2024-06  | Volumetric Representation, Vision-Language Navigation | - | [Paper](http://arxiv.org/abs/2403.14158v1)   | [Code](https://github.com/DefaultRui/VLN-VER) |
| 2024-06  | Precise Manipulation, Few-Shot Learning     | -         | [Paper](http://arxiv.org/abs/2406.08545v1)    | -    |
| 2024-05  | Few-Shot Imitation, Point Tracking         | -         | [Paper](https://arxiv.org/abs/2308.15975)     | -    |
| 2024-02  | 3D Scene Representations, Policy Diffusion | -         | [Paper](http://arxiv.org/abs/2402.10885v3)   | [Code](https://github.com/nickgkan/3d_diffuser_actor) |
| 2023-11  | 3D Feature Fields, Multi-Task Manipulation  | -         | [Paper](http://arxiv.org/abs/2306.17817v2)   | [Code](https://github.com/zhouxian/act3d-chained-diffuser) |
| 2023-11  | Open-World Object Manipulation, Vision-Language Models | - | [Paper](https://arxiv.org/abs/2303.00905)    | -    |
| 2023-11  | Language-Guided Skill Acquisition          | -         | [Paper](http://arxiv.org/abs/2307.14535v2)  | [Code](https://github.com/real-stanford/scalingup) |
| 2023-11  | 3D Value Maps, Language Models             | -         | [Paper](http://arxiv.org/abs/2307.05973v2)   | [Code](https://github.com/huangwl18/VoxPoser) |
| 2023-07  | Visuomotor Policy, Action Diffusion         | -         | [Paper](http://arxiv.org/abs/2303.04137v5)   | [Code](https://github.com/real-stanford/diffusion_policy) |
| 2022-11  | Multi-Task Transformer, Robotic Manipulation | -       | [Paper](http://arxiv.org/abs/2209.05451v2)   | -    |
| 2022-10  | Multimodal Prompts, Robot Manipulation      | -         | [Paper](http://arxiv.org/abs/2210.03094v2)  | -    |
| 2022-07  | Natural Language Feedback, Robot Planning  | -         | [Paper](http://arxiv.org/abs/2204.05186v1)   | -    |
| 2025-01  | Vision-Language Models, Robot Imitation    | -         | [Paper](https://arxiv.org/abs/2311.01378)     | [Code](https://github.com/RoboFlamingo/RoboFlamingo) |
| 2024-10  | 3D Object Understanding, Embodied Interaction | -      | [Paper](http://arxiv.org/abs/2402.17766v3)    | [Code](https://github.com/qizekun/ShapeLLM) |
| 2024-10  | Vision-Language-Action Flow Model          | -         | [Paper](https://arxiv.org/abs/2410.24164)     | -    |
| 2024-07  | Embodied Generalist Agent, 3D World        | -         | [Paper](http://arxiv.org/abs/2311.12871v3)   | [Code](https://github.com/embodied-generalist/embodied-generalist) |
| 2024-07  | Action Hierarchies, Language                | -         | [Paper](http://arxiv.org/abs/2403.01823v2)   | -    |
| 2024-06  | Multisensory Embodied LLM, 3D World        | -         | [Paper](http://arxiv.org/abs/2401.08577v1)    | -    |
| 2024-06  | Spatial Affordance Prediction, Robotics    | -         | [Paper](http://arxiv.org/abs/2406.10721v1)  | [Code](https://github.com/wentaoyuan/RoboPoint) |
| 2024-05  | Skill Induction, Latent Language           | -         | [Paper](http://arxiv.org/abs/2110.01517v2)   | -    |
| 2024-02  | Iterative Visual Prompting, VLMs           | -         | [Paper](http://arxiv.org/abs/2402.07872v1)   | -    |
| 2023-12  | Vision-Language-Action Model, Web Knowledge | -        | [Paper](https://arxiv.org/abs/2307.15818)    | [Code](https://github.com/google-deepmind/open_x_embodiment) |
| 2023-12  | 3D-Enhanced Language Models                | -         | [Paper](http://arxiv.org/abs/2307.12981v1)   | -    |
| 2023-12  | Embodied Chain-of-Thought, Vision-Language | -         | [Paper](http://arxiv.org/abs/2305.15021v2)  | [Code](https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch) |
| 2023-07  | Embodied Multimodal Language Model         | -         | [Paper](http://arxiv.org/abs/2303.03378v1)   | -    |
| 2023-05  | Reasoning-Action Synergy, Language Models  | -         | [Paper](http://arxiv.org/abs/2210.03629v3)  | [Code](https://github.com/ysymyth/ReAct) |
| 2022-12  | Interactive Decision-Making, Language Models | -     | [Paper](https://arxiv.org/abs/2202.01771)    | [Code](https://github.com/ShuangLI59/Pre-Trained-Language-Models-for-Interactive-Decision-Making) |
| 2022-11  | Language Grounding, Robotic Affordances    | -         | [Paper](http://arxiv.org/abs/2204.01691v2)  | [Code](https://github.com/google-research/google-research/tree/master/saycan) |
| 2022-07  | Embodied Reasoning, Language Models        | -         | [Paper](https://arxiv.org/abs/2207.05608)    | -    |
| 2022-07  | Zero-Shot Planning, Embodied Agents        | -         | [Paper](http://arxiv.org/abs/2201.07207v2)   | [Code](https://github.com/huangwl18/language-planner) |
| 2022-05  | Socratic Reasoning, Multimodal Models      | -         | [Paper](http://arxiv.org/abs/2204.00598v2)   | -    |
| 2024-04  | Mobile Manipulation, GPT-4V               | -         | [Paper](https://arxiv.org/abs/2404.10220)     | -    |
| 2023-10  | Grounded Planning, LLMs                    | -         | [Paper](http://arxiv.org/abs/2212.04088v3)   | [Code](https://github.com/OSU-NLP-Group/LLM-Planner/) |
| 2023-06  | ChatGPT Applications, Robotics             | -         | [Paper](http://arxiv.org/abs/2306.17582v2)   | [Code](https://github.com/microsoft/PromptCraft-Robotics) |
| 2023-05  | 3D Scene Graphs, Open-Vocabulary           | -         | [Paper](http://arxiv.org/abs/2309.16650v1)   | [Code](https://github.com/concept-graphs/concept-graphs) |
| 2023-05  | Code Generation, Embodied Control          | -         | [Paper](http://arxiv.org/abs/2209.07753v4)    | [Code](https://github.com/google-research/google-research/tree/master/code_as_policies) |
| 2023-05  | Task Planning, LLMs                        | -         | [Paper](http://arxiv.org/abs/2209.11302v1)   | [Code](https://github.com/NVlabs/progprompt-vh) |
| 2023-02  | Interactive Planning, Multi-Task Agents    | -         | [Paper](http://arxiv.org/abs/2302.01560v3)   | [Code](https://github.com/CraftJarvis/MC-Planner) |


---

## VLA Model Architectures

### General VLAs

| Date       | Keywords          | Institute   |   Paper     |Code      |
| :-----: | :-------------------: | :-------------------: | ---------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------: |
| 2025       | CoT-VLA           | NVIDIA & Stanford    | [Visual Chain-of-Thought Reasoning](http://arxiv.org/abs/2503.22020) | [Code](https://github.com/RoyZry98/MoLe-VLA-Pytorch) |
| 2025       | HybridVLA         | Peking University    | [Collaborative Diffusion and Autoregression](http://arxiv.org/abs/2503.10631) | [Code](https://github.com/PKU-HMI-Lab/Hybrid-VLA) |
| 2025       | DexVLA            | Midea Group          | [Plug-In Diffusion Expert for Robot Control](http://arxiv.org/abs/2502.05855) | [Code](https://github.com/lesjie-wen/dexvla) |
| 2024       | 3D-VLA            | UMass                | [3D Generative World Model](https://arxiv.org/abs/2403.09631) | [Code](https://github.com/UMass-Embodied-AGI/3D-VLA) |
| Mar 27 2025 | CoT-VLA           | NVIDIA & Stanford                                     | CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models | [Code](https://github.com/RoyZry98/MoLe-VLA-Pytorch) |
| Mar 26 2025 | MoLe-VLA          | Nanjing University & HK PolyU & Peking University     | MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation | [Code](https://github.com/RoyZry98/MoLe-VLA-Pytorch) |
| Mar 13 2025 | HybridVLA         | Peking University                                   | HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model | [Code](https://github.com/PKU-HMI-Lab/Hybrid-VLA) |
| Mar 4 2025 | PD-VLA            | HKUST (GZ)                                          | Accelerating Vision-Language-Action Model Integrated with Action Chunking via Parallel Decoding | -                                               |
| Feb 21 2025 | ChatVLA           | Midea Group & East China Normal University        | ChatVLA: Unified Multimodal Understanding and Robot Control with Vision-Language-Action Model | -                                               |
| Feb 21 2025 | VLAS              | Westlake University                               | VLAS: VISION-LANGUAGE-ACTION MODEL WITH SPEECH INSTRUCTIONS FOR CUSTOMIZED ROBOT MANIPULATION | [Code](https://github.com/whichwhichgone/VLAS) |
| Feb 9 2025  | DexVLA            | Midea Group & East China Normal University        | DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control | [Code](https://github.com/lesjie-wen/dexvla) |
| Feb 8 2025  | ConRFT            | Chinese Academy of Sciences                       | ConRFT: A Reinforced Fine-tuning Method for VLA Models via Consistency Policy | -                                               |
| Feb 4 2025  | VLA-Cache         | University of Sydney                              | Towards Efficient Vision-Language-Action Model via Adaptive Token Caching in Robotic Manipulation | -                                               |
| Feb 3 2025  | UP-VLA            | Tsinghua University & Shanghai Qi Zhi Institute   | A Unified Understanding and Prediction Model for Embodied Agent | -                                               |
| Jan 28 2025 | iRe-VLA           | Tsinghua University & Shanghai Qi Zhi Institute   | Improving Vision-Language-Action Model with Online Reinforcement Learning | -                                               |
| 2025-1 | Spatial-VLA       | Shanghai AI Lab                                   | SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Models | [Code](https://github.com/SpatialVLA/SpatialVLA) |
| Jan 16 2025 | FAST              | Physical Intelligence & UC Berkeley & Stanford    | FAST: Efficient Action Tokenization for Vision-Language-Action Models | [Code](https://github.com/Physical-Intelligence/openpi) |
| Jan 8 2025  | FuSe              | UC Berkeley                                       | Beyond Sight: Finetuning Generalist Robot Policies with Heterogeneous Sensors via Language Grounding | [Code](https://github.com/fuse-model/FuSe) |
| Dec 25 2024 | TRACEVLA          | University of Maryland                            | TRACEVLA: VISUAL TRACE PROMPTING ENHANCES SPATIAL-TEMPORAL AWARENESS FOR GENERALIST ROBOTIC POLICIES | -                                               |
| Dec 17 2024 | EMMA-X            | Singapore University of Technology and Design     | EMMA-X: An Embodied Multimodal Action Model with Grounded Chain of Thought and Look-ahead Spatial Reasoning | [Code](https://github.com/declare-lab/Emma-X) |
| Dec 4 2024  | Diffusion-VLA     | East China Normal University                      | Diffusion-VLA: Scaling Robot Foundation Models via Unified Diffusion and Autoregression | -                                               |
| 2024-10 | $\pi_0$           | Physical Intelligence                             | $\pi_0$: A Vision-Language-Action Flow Model for General Robot Control | [Code](https://github.com/Physical-Intelligence/openpi) |
| Jun 13 2024 | OpenVLA           | Stanford University & UC Berkeley & Toyota Research Insititute | OpenVLA: An Open-Source Vision-Language-Action Model | [Code](https://github.com/openvla/openvla) |
| Jul 28 2023 | RT-2              | Google DeepMind                                   | RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control | -                                               |

### Humanoid Robot VLAs


| Date       | Keywords          | Institute   |   Paper     |Code      |
| :-----: | :-------------------: | :-------------------: | ---------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------: |
| 2025       | GR00T N1          | NVIDIA               | [Foundation Model for Humanoid Robots](http://arxiv.org/abs/2503.14734) | [[Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim)] |
| 2025       | Humanoid-VLA      | Westlake University  | [Universal Humanoid Control](http://arxiv.org/abs/2502.14795) | -                                           |
| 2024       | NAVILA            | UC San Diego         | [Legged Robot Navigation](http://arxiv.org/abs/2412.04453) | [Code](https://github.com/navila-bot) |
|2025-3 | GR00T N1          | NVIDIA                                              | GR00T N1: An Open Foundation Model for Generalist Humanoid Robots | [Code](https://github.com/NVIDIA/Isaac-GR00T) |
|2025-3 | GO-1              | AgiBot-World (Shanghai AI Lab & AgiBot Inc.)       | AgiBot World Colosseo: Large-scale Manipulation Platform for Scalable and Intelligent Embodied Systems | [Code](https://github.com/OpenDriveLab/Agibot-World) |
| 2025-2 | Humanoid-VLA      | Westlake University & Zhejiang University           | Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration | -                                               |
|  2024-12  | NAVILA            | UC San Diego                                      | NAVILA: LEGGED ROBOT VISION-LANGUAGEACTION MODEL FOR NAVIGATION | -                                               |
| 2024-05 | OTTER             | UC Berkeley                                       | [OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature Extraction](https://arxiv.org/abs/2503.03734) | [Code JAX](https://github.com/FangchenLiu/otter_jax) |
| 2024-05 | Octo              | UC Berkeley                                       | Octo: An Open-Source Generalist Robot Policy | [Code](https://github.com/octo-models/octo) |

### Robotic Foundation Models


| Date       | Keywords          | Institute   |   Paper     |Code      |
| :-----: | :-------------------: | :-------------------: | ---------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------: |
| 2024       | Octo        | UC Berkeley   | [Open-Source Generalist Policy](https://arxiv.org/abs/2405.12213) | [Code](https://github.com/octo-models/octo) |
| 2025       | OTTER       | UC Berkeley   | [Text-Aware Visual Features](http://arxiv.org/abs/2503.03734) | [Code](https://github.com/FangchenLiu/otter_jax) |
| Mar 5 2025 | OTTER             | UC Berkeley | OTTER: A Vision-Language-Action Model with Text-Aware Visual Feature Extraction | [[Code JAX](https://github.com/FangchenLiu/otter_jax)] |
| May 20 2024 | Octo              | UC Berkeley | Octo: An Open-Source Generalist Robot Policy    | [Code](https://github.com/octo-models/octo)       |

---

## Components of VLA Systems

### Reinforcement Learning

| Date       | Keywords          | Institute   |   Paper     |Code      |
| :-----: | :-------------------: | :-------------------: | ---------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------: |
| 2021       | Decision Transformer | NeurIPS     | [RL via Sequence Modeling](https://arxiv.org/abs/2106.01345) | [Code](https://github.com/kzl/decision-transformer) |
| 2023       | Reflexion          | NeurIPS     | [Language Agents with Verbal RL](https://arxiv.org/abs/2303.11366) | [Code](https://github.com/noahshinn/reflexion) |

### Pretrained Visual Representations


| Date       | Keywords          | Institute   |   Paper     |Code      |
| :-----: | :-------------------: | :-------------------: | ---------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------: |
| 2021       | CLIP              | OpenAI      | [Transferable Visual-Language Models](https://arxiv.org/abs/2103.00020) | [Code](https://github.com/openai/CLIP) |
| 2023       | DINOv2            | Meta        | [Self-Supervised Visual Features](https://arxiv.org/abs/2304.07193) | [Code](https://github.com/facebookresearch/dinov2) |

### World Models

| Date       | Keywords          | Institute   |   Paper     |Code      |
| :-----: | :-------------------: | :-------------------: | ---------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------: |
| 2023       | DreamerV3         | -           | [Scalable World Models](https://arxiv.org/abs/2301.04104) | -               |
| 2024       | Genie             | Google      | [Generative Interactive Environments](https://arxiv.org/abs/2402.15391) | [[Demo](https://sites.google.com/view/genie-2024)] |

---

## Datasets & Benchmarks

|  Date   |   keywords    | Institute | Paper                                                                                                   |                                           Code                                            |
| :-----: | :-------------------: | :-------------------: | ---------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------: |
| 2023-10 | OpenX Dataset |     -     | [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/pdf/2310.08864)        | [google-deepmind/open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment) |
| 2024-08 | Data Mixtures | Stanford  | [Re-Mix: Optimizing Data Mixtures for Large Scale Imitation Learning](https://arxiv.org/pdf/2408.14037) |                      [jhejna/remix](https://github.com/jhejna/remix)                      |
| 2024-05 |   Real2Sim    |   UCSD    | [Evaluating Real-World Robot Manipulation Policies in Simulation](https://arxiv.org/abs/2405.05941)     |            [simpler-env/SimplerEnv](https://github.com/simpler-env/SimplerEnv)            |
| 2023-06 |   Lifelong    |    UT     | [Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://arxiv.org/abs/2306.03310)         |    [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)    |

---

## Tutorials & Courses


| Date       | Keywords          | Institute   |   Paper     |Code      |
| :-----: | :-------------------: | :-------------------: | ---------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------: |
| 2024       | CoRL Tutorial     | -           | -           | [[Video](https://www.bilibili.com/video/BV1p3UYYxEFb)] |
| 2024-08         | OpenVLA Talk      | -           | -           | [[YouTube](https://www.youtube.com/watch?v=-0s0v3q7mBk)] |
| -          | Diffusion Policy  | -           | -           | [[Code](https://github.com/real-stanford/diffusion_policy)] |
| 2024-11     | From Octo to π₀: How to Train Your Generalist Robot Policy  | -           | -           |[CoRL24-8 From Octo to π₀: How to Train Your Generalist Robot Policy](https://www.bilibili.com/video/BV1p3UYYxEFb/?share_source=copy_web&vd_source=7b9c04cb5a01c024b1b34f587bb769ce)|
| 2024-10     | RDT-1B Talk  | -           | -           |[RDT-1B Talk](https://www.bilibili.com/video/BV1FjyHYmEDQ/?share_source=copy_web&vd_source=7b9c04cb5a01c024b1b34f587bb769ce)|






---

## Related Surveys


| Date       | Keywords          | Institute   |   Paper     |Code      |
| :-----: | :-------------------: | :-------------------: | ---------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------: |
| 2023       | Foundation Models in Robotics  | -         | [Paper](https://arxiv.org/abs/2312.07843) | -    |
| 2024       | Large Language Models for Robotics | -       | [Paper](https://arxiv.org/abs/2401.04334) | -    |
| 2024       | A Survey on VLA Models for Embodied AI | CUHK    | [Paper](https://arxiv.org/abs/2405.14093) | -    |
| 2024-05   | World Models, Sora             | -         | [Paper](http://arxiv.org/abs/2405.03520v1)    | -    |
| 2024-02   | Foundation Models, Robotics    | -         | [Paper](http://arxiv.org/abs/2402.05741v2)   | -    |
| 2024-01   | LLMs, Robotics                 | -         | [Paper](https://arxiv.org/abs/2401.04334)    | -    |
| 2023-12   | Foundation Models, Robotics    | -         | [Paper](http://arxiv.org/abs/2312.07843v1)   | -    |
| 2023-12   | Foundation Models, General-Purpose Robots | - | [Paper](http://arxiv.org/abs/2312.08782v3) | - |
---

## Contributing
Contributions are welcome! Please submit pull requests or contact [yuyinfeng@xju.edu.cn](mailto:yuyinfeng@xju.edu.cn) for suggestions.  
*Maintained by Yinfeng Yu  and collaborators.*  
