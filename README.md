# Flask Micro Alpha Trader

**Flask Micro Alpha Trader**는 AI/ML 기반의 자동화된 주식 거래 시스템입니다. Alpaca API를 통해 실시간 데이터를 수집하고, 앙상블 머신러닝 모델로 거래 신호를 생성하여 자동으로 거래를 실행합니다. 독립적인 서비스 모듈과 비동기 작업 처리(Celery)에 중점을 둔 확장 가능한 모놀리식 아키텍처(Scalable Monolithic Architecture)**로 설계했습니다.

## 🚀 주요 기능

### 🤖 AI 기반 트레이딩
- **앙상블 모델**: LightGBM과 XGBoost를 결합한 앙상블 방식으로 예측 신뢰도 향상
- **동적 신뢰도 계산**: 모델 예측값의 표준편차를 기반으로 신뢰도 동적 계산
- **자동 하이퍼파라미터 최적화**: Optuna를 사용한 주기적 파라미터 튜닝
- **실시간 예측**: 120일간의 과거 데이터를 기반으로 한 일일 수익률 예측

### 🔄 자동화 트레이딩 파이프라인
- **주기적 실행**: Celery Beat를 통한 5분마다 자동 트레이딩 실행
- **스마트 신호 생성**: AI 신뢰도와 예측 수익률을 기반으로 한 BUY/SELL/HOLD 신호
- **포트폴리오 리밸런싱**: 새로운 기회 발견 시 기존 포지션 교체 로직
- **손절/익절 자동화**: 설정된 손실(-5%) 및 이익(+10%) 기준에 따른 자동 청산

### 🛡️ 고급 리스크 관리
- **변동성 기반 포지션 크기**: 시장 변동성에 따른 동적 투자 규모 조절
- **일일 손실 제한**: 일일 최대 손실(-2%) 도달 시 24시간 거래 중단
- **포트폴리오 제약**: 최대 7개 포지션, 총 포트폴리오의 80% 이하 투자
- **실시간 리스크 모니터링**: JSON 파일 기반 상태 관리로 재시작 후에도 리스크 상태 유지

### 📊 모니터링 및 분석
- **실시간 대시보드**: Grafana를 통한 포트폴리오, AI 성능, 시스템 메트릭 시각화
- **백테스팅 엔진**: 과거 데이터로 전략 성능 검증
- **거래 이력 추적**: SQLite 기반 거래 기록 및 성과 분석
- **Discord 알림**: 주요 거래 및 시스템 이벤트 실시간 알림

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────┐    ┌─────────────────────────┐    ┌─────────────────────────┐
│        Flask API        │    │      Celery Worker      │    │       Celery Beat       │
│    (REST Endpoints)     │    │    (Background Jobs)    │    │        (Scheduler)      │
└─────────────────────────┘    └─────────────────────────┘    └─────────────────────────┘
             │                             │                             │
             └─────────────────────────────┼─────────────────────────────┘
                                           │
         ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
         │       Redis       │    │    Prometheus     │    │      Grafana      │
         │ (Message Broker)  │    │     (Metrics)     │    │    (Dashboard)    │
         └───────────────────┘    └───────────────────┘    └───────────────────┘
```


### 핵심 서비스 모듈
- **AlpacaService**: Alpaca API 통신 및 시장 데이터 수집
- **AIService**: 앙상블 ML 모델 관리 및 신호 생성
- **TradingService**: 주문 실행 및 포트폴리오 관리
- **RiskManager**: 실시간 리스크 모니터링 및 제어
- **DatabaseService**: 거래 이력 및 성과 데이터 관리
- **BacktestService**: 전략 백테스팅 및 성과 분석

## 🛠️ 기술 스택

- **Backend**: Flask, Gunicorn, Celery
- **AI/ML**: LightGBM, XGBoost, Optuna, NumPy, Pandas
- **Database**:
      SQLite: 거래 데이터 기록 (프로토타이핑 단계에서의 빠른 개발 및 로컬 테스트 용이성을 위해 선택)
      JSON File: 리스크 상태 관리 (외부 DB 의존성 없이 상태를 빠르게 백업 및 복구하는 프로토타입 용도로 사용)
      Note: 프로덕션 환경에서는 동시성 처리 및 데이터 무결성을 위해 **PostgreSQL/MySQL(거래 데이터)**과 **Redis(실시간 상태 관리)**로 전환
- **Message Queue**: Redis
- **Monitoring**: Prometheus, Grafana, prometheus-flask-exporter
- **External API**: Alpaca Markets API
- **Containerization**: Docker, Docker Compose

## 📈 지원 종목

**기본 관심종목 (32개)**:
```
AAPL, MSFT, GOOGL, TSLA, SPY, NVDA, AMZN, META, NFLX, AMD, 
JNJ, UNH, JPM, PG, CAT, XOM, V, MA, DIS, HD, KO, PEP, 
INTC, CSCO, CMCSA, VZ, T, MRK, PFE, ABT, NKE, WMT
```

## 🚀 시작하기

### 사전 요구사항
- [Docker](https://docs.docker.com/get-docker/) 및 [Docker Compose](https://docs.docker.com/compose/install/)
- [Alpaca API 계정](https://alpaca.markets/) (Paper Trading 또는 Live Trading)
- Discord 웹훅 URL (선택사항, 알림용)

### 설치 및 실행

1. **저장소 클론**
   ```bash
   git clone https://github.com/your-username/flask-micro-alpha-trader.git
   cd flask-micro-alpha-trader
   ```

2. **환경 변수 설정**
   ```bash
   cp .env.example .env
   nano .env
   ```

   **필수 환경 변수**:
   ```env
   # Alpaca API 설정
   ALPACA_API_KEY=your_alpaca_api_key
   ALPACA_API_SECRET=your_alpaca_secret_key
   ALPACA_PAPER=True  # Paper Trading 사용 여부

   # Discord 알림 (선택사항)
   DISCORD_WEBHOOK_URL=your_discord_webhook_url

   # JWT 및 보안
   SECRET_KEY=your-very-secure-secret-key

   # Redis 설정
   REDIS_URL=redis://redis:6379/0
   ```

3. **전체 시스템 시작**
   ```bash
   docker-compose up --build -d
   ```

4. **로그 확인**
   ```bash
   docker-compose logs -f
   ```

### 서비스 접속 정보

| 서비스 | URL | 설명 |
|--------|-----|------|
| **API Server** | http://localhost:5000 | REST API 엔드포인트 |
| **Grafana** | http://localhost:3000 | 모니터링 대시보드 (admin/admin) |
| **Flower** | http://localhost:5555 | Celery 작업 모니터링 |
| **Prometheus** | http://localhost:9090 | 메트릭 수집 서버 |

## 🔧 주요 API 엔드포인트

### 인증 (JWT 토큰을 위한 임시 인증)
```bash
# 로그인 (기본 계정: admin/password)
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'
```

### 계정 및 포지션 조회
```bash
# JWT 토큰을 Authorization 헤더에 포함
AUTH_HEADER="Authorization: Bearer YOUR_JWT_TOKEN"

# 계정 정보 조회
curl -H "$AUTH_HEADER" http://localhost:5000/api/account/info

# 현재 포지션 조회
curl -H "$AUTH_HEADER" http://localhost:5000/api/positions

# 특정 종목 현재가 조회
curl -H "$AUTH_HEADER" http://localhost:5000/api/price/AAPL
```

### AI 및 트레이딩
```bash
# 특정 종목 AI 신호 조회
curl -H "$AUTH_HEADER" http://localhost:5000/api/trading/signal/AAPL

# 트레이딩 파이프라인 수동 실행
curl -H "$AUTH_HEADER" http://localhost:5000/api/trading/auto/start

# 시장 데이터 조회
curl -H "$AUTH_HEADER" "http://localhost:5000/api/market/bars/AAPL?limit=10"
```

### 모델 학습 및 최적화
```bash
# 전체 종목 모델 학습 시작
curl -X POST -H "$AUTH_HEADER" http://localhost:5000/api/training/start/batch

# 하이퍼파라미터 최적화 시작
curl -X POST -H "$AUTH_HEADER" http://localhost:5000/api/hyperparameter/optimize/

# 백테스팅 실행
curl -X POST -H "$AUTH_HEADER" http://localhost:5000/api/backtest/run \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL", "MSFT"], "days": 90}'
```

### 리스크 및 대시보드
```bash
# 리스크 상태 조회
curl -H "$AUTH_HEADER" http://localhost:5000/api/risk/status

# 대시보드 통합 데이터
curl -H "$AUTH_HEADER" http://localhost:5000/api/dashboard/all_data
```

## ⚙️ 설정 관리

### 리스크 관리 설정 (`app/config.py`)
```python
# 포트폴리오 제한
RISK_MAX_PORTFOLIO_EXPOSURE = 0.80    # 최대 80% 투자
RISK_MAX_POSITIONS = 7                # 최대 7개 포지션
RISK_MAX_POSITION_SIZE = 0.10         # 개별 포지션 최대 10%

# 손익 기준
RISK_STOP_LOSS_PCT = -0.05           # -5% 손절
RISK_TAKE_PROFIT_PCT = 0.10          # +10% 익절
RISK_MAX_DAILY_LOSS = -0.02          # 일일 최대 -2% 손실

# AI 신호 임계값
AI_SIGNAL_BUY_THRESHOLD = 0.005      # 매수 신호: +0.5% 이상 예측
AI_SIGNAL_SELL_THRESHOLD = -0.005    # 매도 신호: -0.5% 이하 예측
AI_CONFIDENCE_THRESHOLD = 0.7        # 최소 신뢰도 70%
```

### Celery 스케줄 (`app/celery_config.py`) 
# UTC 기준
```python
beat_schedule = {
    'smart-trading-pipeline': {
        'task': 'app.tasks.smart_trading_pipeline',
        'schedule': crontab(minute='*/10', hour='11-23', day_of_week='mon-fri'),  
    },
    'daily-model-finetuning': {
        'task': 'app.tasks.train_models_batch',
        'schedule': crontab(hour=23, minute=0, day_of_week='mon-fri'),
    },
    'optimize-hyperparameters': {
        'task': 'app.tasks.optimize_hyperparameters_for_watchlist',
        'schedule': crontab(hour=23, minute=0, day_of_week='mon-fri'),
    }
}
```

## 📊 모니터링

### Grafana 대시보드
시스템 시작 후 http://localhost:3000 접속 (admin/admin)

**주요 메트릭**:
- 포트폴리오 가치 추이
- 일일 거래량 및 수익률
- AI 모델 신뢰도
- 시스템 리소스 사용률
- Celery 작업 상태

### 로그 모니터링
```bash
# 실시간 로그 확인
docker-compose logs -f flask
docker-compose logs -f celery_worker
docker-compose logs -f celery_beat

# 특정 서비스 로그
docker logs alpha-trader-flask
```

## 🧪 백테스팅

시스템에 내장된 백테스팅 엔진을 사용하여 전략 성능을 검증할 수 있습니다:

```bash
# 90일간 AAPL, MSFT 백테스팅
curl -X POST -H "$AUTH_HEADER" http://localhost:5000/api/backtest/run \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["AAPL", "MSFT"], 
    "days": 90,
    "initial_capital": 100000
  }'
```

**백테스팅 결과 예시**:
```json
{
  "status": "success",
  "performance": {
    "total_return": 0.1234,
    "annualized_return": 0.4567,
    "sharpe_ratio": 1.23,
    "max_drawdown": -0.0567,
    "win_rate": 0.65
  }
}
```

## 🔒 보안 고려사항

1. **API 키 보안**: `.env` 파일을 절대 공개 저장소에 커밋하지 마세요
2. **Paper Trading**: 실제 자금 투입 전 Paper Trading으로 충분한 테스트 수행
3. **JWT 토큰**: 기본 24시간 만료, 프로덕션에서는 더 짧게 설정 권장
4. **방화벽**: 프로덕션 환경에서는 불필요한 포트 차단

## 🐛 트러블슈팅

### 자주 발생하는 문제

1. **Alpaca API 연결 오류**
   ```bash
   # API 키 확인
   curl -H "$AUTH_HEADER" http://localhost:5000/api/test/connection
   ```

2. **모델 파일 없음 오류**
   ```bash
   # 모델 학습 실행
   curl -X POST -H "$AUTH_HEADER" http://localhost:5000/api/training/start/batch
   ```

3. **메모리 부족**
   ```bash
   # Docker 리소스 확인
   docker stats
   # 컨테이너 재시작
   docker-compose restart
   ```

4. **Celery 작업 실패**
   ```bash
   # Celery 상태 확인
   docker-compose logs celery_worker
   # Flower에서 상세 확인: http://localhost:5555
   ```

## 📁 프로젝트 구조

```
flask-micro-alpha-trader/
├── app/                          # Flask 애플리케이션
│   ├── services/                 # 비즈니스 로직 서비스
│   │   ├── alpaca_service.py     # Alpaca API 통신
│   │   ├── ai_service.py         # AI 모델 관리
│   │   ├── trading_service.py    # 거래 실행
│   │   ├── risk_manager.py       # 리스크 관리
│   │   ├── database_service.py   # 데이터베이스 관리
│   │   ├── backtest_service.py   # 백테스팅 엔진
│   │   └── optimizer_service.py  # 하이퍼파라미터 최적화
│   ├── __init__.py              # Flask 앱 팩토리 및 라우트
│   ├── tasks.py                 # Celery 백그라운드 작업
│   ├── celery_config.py         # Celery 스케줄 설정
│   └── config.py                # 환경별 설정
├── models/                      # 학습된 AI 모델 (자동 생성)
├── best_params/                 # 최적화된 하이퍼파라미터 (자동 생성)
├── database/                    # SQLite 데이터베이스 (자동 생성)
├── grafana/                     # Grafana 설정
│   └── provisioning/            # 대시보드 및 데이터소스
├── logs/                        # 애플리케이션 로그 (자동 생성)
├── docker-compose.yml           # 서비스 오케스트레이션
├── Dockerfile                   # 애플리케이션 이미지
├── requirements.txt             # Python 의존성
├── prometheus.yml               # Prometheus 설정
├── gunicorn_config.py          # Gunicorn 설정
└── .env.example                # 환경 변수 템플릿
```

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## ⚠️ 면책 조항

이 소프트웨어는 교육 및 연구 목적으로만 제공됩니다. 실제 거래에서 발생하는 손실에 대해 개발자는 책임지지 않습니다. 투자는 본인의 판단과 책임 하에 이루어져야 합니다.
