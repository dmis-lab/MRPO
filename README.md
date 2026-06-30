# Breaking Failure Cascades: Step-Aware Reinforcement Learning for Medical Multimodal Reasoning

<!-- <p align="center">
        🤗 <a href="https://huggingface.co/papers/2601.05242">Hugging Face Page</a>&nbsp&nbsp | &nbsp&nbsp 📄 <a href="https://arxiv.org/abs/2601.05242">Paper</a> | &nbsp&nbsp 📜 <a href="https://nvlabs.github.io/GDPO/">Page</a> &nbsp
</p> -->

### News
* [June/30/2026] 🎉 We release our paper "Breaking Failure Cascades: Step-Aware Reinforcement Learning for Medical Multimodal Reasoning" on arXiv.


## Overview
MRPO is a novel reinforcement learning framework that improves medical multimodal reasoning by directly addressing failures in the reasoning process. It reshapes GRPO-style advantages using both answer-level and step-wise process rewards, assigning exponentially larger penalties to earlier invalid steps when the final answer is incorrect, thereby correcting early-stage failures before they cascade while preserving successful trajectories. By redistributing the learning signal according to where reasoning first fails, MRPO induces transferable reasoning that improves both reasoning quality and final answer accuracy across diverse medical VQA benchmarks.