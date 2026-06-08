---
title: Prometheus를 활용한 메트릭 모니터링
description: Prometheus로 시스템 및 서비스 메트릭을 수집하고 모니터링하는 방법
author: hchvhp1543
date: 2026-05-13 22:00:00 +0900
categories: [Observability, Metrics]
tags: [prometheus, metrics, monitoring]
---

## Prometheus를 활용한 메트릭 모니터링

### 소개
Prometheus는 SoundCloud에서 최초로 개발한 오픈 소스 시스템 모니터링 및 경고(Alerting) 툴킷입니다. 현재는 클라우드 네이티브 컴퓨팅 재단(CNCF)의 졸업 프로젝트로서 독립적으로 널리 사용되고 있습니다.

### 주요 기능
- **다차원 데이터 모델:** 메트릭 이름과 키/값 쌍(레이블)으로 식별되는 시계열 데이터 모델.
- **PromQL:** 다차원 데이터를 조회하고 정밀하게 분석하기 위한 유연한 쿼리 언어 제공.
- **Pull 모델:** HTTP를 통해 주기적으로 메트릭 데이터를 긁어오는(Scraping) 방식 채택.
- **서비스 디스커버리:** Kubernetes 등 다양한 클라우드 환경에서 모니터링 대상을 자동으로 탐지.
