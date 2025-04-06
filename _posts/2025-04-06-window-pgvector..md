---
title: Window 환경에서 pgvector 설치해 사용하기
description: WSL, Docker, Visual Studio Build Tools 설치가 안되는 환경에서 pgvector를 설치해 사용해보자
author: hchvhp1543
date: 2025-04-06 11:33:00 +0900
categories: [DEV, Data]
tags: [window, postgresql, pgvector, vector_db]
pin: false
math: true
mermaid: true

---

# Background

* 회사에서 개발용으로 받은 윈도우 노트북. 
* 여러가지 제약 존재
  * WSL 설치 불가
  * Docker 사용 불가
  * visual studio (build tools) 빌드 도구도 설치 불가 (enterprise 라이센스 필요)
* pgvector 관련 기능 검토 필요.
  * 윈도우에서 pgvector 설치시 다음 조건 필요
    > Ensure C++ support in Visual Studio is installed
  * **다행히 conda-forge 통해 설치할수 있다는 가이드 존재**
    

# prerequisites
* 윈도우 10
* git-bash 설치
* miniconda 설치
* postgresql 설치
* db 접속 프로그램 설치 : dbeaver 등


# Process
* 가상 환경 생성
```bash
conda create -n pgvector python=3.12
conda activate pgvector
```
* conda-forge 통한 설치
```bash
conda install -c conda-forge pgvector
```
