---
title: OpenTelemetry (OTel) Collector 소개
description: OpenTelemetry Collector가 어떻게 텔레메트리 데이터의 프록시 역할을 하는지 알아봅니다.
author: hchvhp1543
date: 2026-04-22 22:00:00 +0900
categories: [Observability, OpenTelemetry]
tags: [otel, telemetry, collector]
---

## OpenTelemetry (OTel) Collector 소개

### 소개
OpenTelemetry Collector는 특정 벤더에 종속되지 않고 텔레메트리 데이터(메트릭, 로그, 트레이스)를 수집, 처리 및 내보내는(export) 아키텍처를 제공합니다. 이를 통해 여러 종류의 에이전트나 수집기를 개별적으로 운영 및 유지 관리해야 하는 번거로움을 줄여줍니다.

### 구성 요소
- **Receivers (수신기):** Collector로 데이터를 가져오는 방식 정의 (예: OTLP, Prometheus).
- **Processors (처리기):** 데이터의 가공, 필터링, 배치 처리를 담당.
- **Exporters (내보내기):** 가공된 데이터를 외부 백엔드 시스템(예: Jaeger, Prometheus, Loki)으로 전송.
