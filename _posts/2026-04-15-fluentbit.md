---
title: Fluent Bit을 이용한 로그 수집
description: Fluent Bit 소개와 이를 활용하여 로그를 수집하는 방법
author: hchvhp1543
date: 2026-04-15 22:00:00 +0900
categories: [Observability, Logging]
tags: [fluentbit, logging, observability]
---

## Fluent Bit을 이용한 로그 수집

### 소개
Fluent Bit은 매우 빠르고 가벼우며 확장성이 뛰어난 로그 및 메트릭 프로세서이자 포워더입니다. 다양한 소스로부터 데이터를 수집하고 이를 통합하여 여러 목적지로 안전하게 전송할 수 있도록 설계되었습니다.

### 주요 기능
- **고성능:** C 언어로 작성되어 메모리 사용량이 매우 적습니다.
- **플러그인 아키텍처:** 풍부한 입력(Inputs), 필터(Filters), 출력(Outputs) 플러그인을 제공합니다.
- **쿠버네티스 통합:** 쿠버네티스 메타데이터를 수집하기 위한 네이티브 지원을 포함합니다.
