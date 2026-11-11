# RuralLens App

RuralLens offline visual triage app source and debug Android build (snapshot 2026-10-22).

Contents: `src/App.jsx`, `data/diseases.json`, `dist/index.html`, and the debug APK plus its recorded SHA-256.

## Scope and limitations

This snapshot is an offline, Africa-focused visual-triage research prototype, not a validated diagnostic system. The included partial-domain checkpoints do **not** meet the project's 90%+ leakage-safe held-out-accuracy gate for all 20 classes in each of crop, animal, and human-skin domains. No valid full 20-class held-out accuracy is reported. Read the audit and boundary reports before interpreting or reusing any artifacts.

Some upstream image sources are documented as non-commercial and/or no-derivatives, and image-level attribution requirements apply. The relevant source-license notes are included in this repository. Do not assume the model weights or downstream application can be used commercially without reviewing those restrictions and the source dataset licenses. No license is granted here beyond rights the uploader has to share these materials.

The human-skin user experience must retain the visible notice: “Human-skin results are not a medical diagnosis.” The app is for educational triage only; it cannot confirm a disease, toxin, or treatment.


The APK is a debug build, not a production or medically validated release. This repository does not include a root dependency manifest/build configuration, so the included source/static snapshot should not be treated as independently reproducible without the companion engine/RAG repositories and missing project build files.

## Checksum

The checksum file is `artifacts/rurallens-debug.apk.sha256`.
