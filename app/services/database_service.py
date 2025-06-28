import sqlite3
import json
import os
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
from flask import current_app

class DatabaseService:    
    def __init__(self, db_path: str = "database/trades.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 거래 테이블
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    order_id TEXT UNIQUE,
                    executed_at TIMESTAMP NOT NULL,
                    ai_signal TEXT,
                    ai_confidence REAL,
                    ai_predicted_return REAL,
                    commission REAL DEFAULT 0.0,
                    status TEXT DEFAULT 'executed',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    total_quantity INTEGER NOT NULL,
                    avg_cost_basis REAL NOT NULL,
                    total_invested REAL NOT NULL,
                    first_buy_date TIMESTAMP NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_date DATE UNIQUE NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    buy_trades INTEGER DEFAULT 0,
                    sell_trades INTEGER DEFAULT 0,
                    total_volume REAL DEFAULT 0.0,
                    realized_pnl REAL DEFAULT 0.0,
                    portfolio_value_start REAL,
                    portfolio_value_end REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(executed_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol)')
            
            conn.commit()
            conn.close()
            
            current_app.logger.info("거래 추적 데이터베이스 초기화 완료")
            
        except Exception as e:
            current_app.logger.error(f"데이터베이스 초기화 실패: {e}")
            raise
    
    def record_trade(self, trade_data: Dict) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    symbol, side, quantity, price, order_id, executed_at,
                    ai_signal, ai_confidence, ai_predicted_return, commission, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data['symbol'],
                trade_data['side'],
                trade_data['quantity'],
                trade_data['price'],
                trade_data.get('order_id'),
                trade_data.get('executed_at', datetime.now()),
                trade_data.get('ai_signal'),
                trade_data.get('ai_confidence'),
                trade_data.get('ai_predicted_return'),
                trade_data.get('commission', 0.0),
                trade_data.get('status', 'executed')
            ))
            
            self._update_position(cursor, trade_data)
            
            self._update_daily_stats(cursor, trade_data)
            
            conn.commit()
            conn.close()
            
            current_app.logger.info(f"거래 기록 완료: {trade_data['symbol']} {trade_data['side']} {trade_data['quantity']}주")
            return True
            
        except Exception as e:
            current_app.logger.error(f"거래 기록 실패: {e}")
            return False
    
    def _update_position(self, cursor: sqlite3.Cursor, trade_data: Dict):
        symbol = trade_data['symbol']
        side = trade_data['side']
        quantity = trade_data['quantity']
        price = trade_data['price']
        
        cursor.execute('SELECT * FROM positions WHERE symbol = ?', (symbol,))
        position = cursor.fetchone()
        
        if side == 'BUY':
            if position:
                current_qty = position[2]
                current_avg_cost = position[3]
                current_invested = position[4]
                
                new_qty = current_qty + quantity
                new_invested = current_invested + (quantity * price)
                new_avg_cost = new_invested / new_qty
                
                cursor.execute('''
                    UPDATE positions 
                    SET total_quantity = ?, avg_cost_basis = ?, total_invested = ?, last_updated = ?
                    WHERE symbol = ?
                ''', (new_qty, new_avg_cost, new_invested, datetime.now(), symbol))
            else:
                cursor.execute('''
                    INSERT INTO positions (symbol, total_quantity, avg_cost_basis, total_invested, first_buy_date)
                    VALUES (?, ?, ?, ?, ?)
                ''', (symbol, quantity, price, quantity * price, datetime.now()))
        
        elif side == 'SELL' and position:
            current_qty = position[2]
            current_invested = position[4]
            
            if quantity >= current_qty:
                cursor.execute('DELETE FROM positions WHERE symbol = ?', (symbol,))
            else:
                new_qty = current_qty - quantity
                sell_ratio = quantity / current_qty
                new_invested = current_invested * (1 - sell_ratio)
                
                cursor.execute('''
                    UPDATE positions 
                    SET total_quantity = ?, total_invested = ?, last_updated = ?
                    WHERE symbol = ?
                ''', (new_qty, new_invested, datetime.now(), symbol))
    
    def _update_daily_stats(self, cursor: sqlite3.Cursor, trade_data: Dict):
        trade_date = date.today()
        side = trade_data['side']
        volume = trade_data['quantity'] * trade_data['price']
        
        cursor.execute('SELECT * FROM daily_stats WHERE trade_date = ?', (trade_date,))
        stats = cursor.fetchone()
        
        if stats:
            total_trades = stats[2] + 1
            buy_trades = stats[3] + (1 if side == 'BUY' else 0)
            sell_trades = stats[4] + (1 if side == 'SELL' else 0)
            total_volume = stats[5] + volume
            
            cursor.execute('''
                UPDATE daily_stats 
                SET total_trades = ?, buy_trades = ?, sell_trades = ?, total_volume = ?
                WHERE trade_date = ?
            ''', (total_trades, buy_trades, sell_trades, total_volume, trade_date))
        else:
            cursor.execute('''
                INSERT INTO daily_stats (trade_date, total_trades, buy_trades, sell_trades, total_volume)
                VALUES (?, 1, ?, ?, ?)
            ''', (trade_date, 1 if side == 'BUY' else 0, 1 if side == 'SELL' else 0, volume))
    
    def get_trading_summary(self, days: int = 30) -> Dict:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN side = 'BUY' THEN 1 ELSE 0 END) as buy_trades,
                    SUM(CASE WHEN side = 'SELL' THEN 1 ELSE 0 END) as sell_trades,
                    SUM(quantity * price) as total_volume,
                    AVG(ai_confidence) as avg_confidence
                FROM trades 
                WHERE executed_at >= date('now', '-{} days')
            '''.format(days))
            
            summary = cursor.fetchone()
            
            cursor.execute('''
                SELECT COUNT(*) as position_count, SUM(total_invested) as total_invested
                FROM positions
            ''')
            
            positions = cursor.fetchone()
            
            realized_pnl = self._calculate_realized_pnl(cursor, days)
            
            win_rate = self._calculate_win_rate(cursor, days)

            conn.close()
            
            return {
                "period_days": days,
                "total_trades": summary[0] or 0,
                "buy_trades": summary[1] or 0,
                "sell_trades": summary[2] or 0,
                "total_volume": summary[3] or 0.0,
                "avg_ai_confidence": summary[4] or 0.0,
                "current_positions": positions[0] or 0,
                "total_invested": positions[1] or 0.0,
                "realized_pnl": realized_pnl,
                "win_rate": win_rate
            }
            
        except Exception as e:
            current_app.logger.error(f"거래 요약 조회 실패: {e}")
            return {}
    
    def _calculate_realized_pnl(self, cursor: sqlite3.Cursor, days: int) -> float:
        try:
            cursor.execute('''
                SELECT symbol, quantity, price, executed_at
                FROM trades 
                WHERE side = 'SELL' AND executed_at >= date('now', '-{} days')
                ORDER BY symbol, executed_at
            '''.format(days))
            
            sells = cursor.fetchall()
            total_pnl = 0.0
            
            for sell in sells:
                symbol, sell_qty, sell_price, sell_date = sell
                
                cursor.execute('''
                    SELECT quantity, price
                    FROM trades 
                    WHERE symbol = ? AND side = 'BUY' AND executed_at <= ?
                    ORDER BY executed_at
                ''', (symbol, sell_date))
                
                buys = cursor.fetchall()
                
                remaining_sell_qty = sell_qty
                for buy_qty, buy_price in buys:
                    if remaining_sell_qty <= 0:
                        break
                    
                    matched_qty = min(remaining_sell_qty, buy_qty)
                    pnl = matched_qty * (sell_price - buy_price)
                    total_pnl += pnl
                    
                    remaining_sell_qty -= matched_qty
            
            return total_pnl
            
        except Exception as e:
            current_app.logger.error(f"실현 손익 계산 실패: {e}")
            return 0.0
    
    def _calculate_win_rate(self, cursor: sqlite3.Cursor, days: int) -> float:
        try:
            cursor.execute('''
                SELECT symbol, quantity, price, executed_at
                FROM trades 
                WHERE side = 'SELL' AND executed_at >= date('now', '-{} days')
            '''.format(days))
            
            sells = cursor.fetchall()
            winning_trades = 0
            total_trades = len(sells)
            
            if total_trades == 0:
                return 0.0
            
            for sell in sells:
                symbol, sell_qty, sell_price, sell_date = sell
                
                cursor.execute('''
                    SELECT AVG(price) as avg_buy_price
                    FROM trades 
                    WHERE symbol = ? AND side = 'BUY' AND executed_at <= ?
                ''', (symbol, sell_date))
                
                result = cursor.fetchone()
                if result and result[0]:
                    avg_buy_price = result[0]
                    if sell_price > avg_buy_price:
                        winning_trades += 1
            
            return winning_trades / total_trades if total_trades > 0 else 0.0
            
        except Exception as e:
            current_app.logger.error(f"승률 계산 실패: {e}")
            return 0.0
    
    def get_daily_performance(self, days: int = 7) -> List[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT 
                    trade_date,
                    total_trades,
                    buy_trades,
                    sell_trades,
                    total_volume,
                    realized_pnl,
                    portfolio_value_end
                FROM daily_stats 
                WHERE trade_date >= date('now', '-{} days')
                ORDER BY trade_date DESC
            '''.format(days))
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "date": row[0],
                    "total_trades": row[1],
                    "buy_trades": row[2],
                    "sell_trades": row[3],
                    "volume": row[4],
                    "realized_pnl": row[5] or 0.0,
                    "portfolio_value": row[6]
                }
                for row in results
            ]
            
        except Exception as e:
            current_app.logger.error(f"일별 성과 조회 실패: {e}")
            return []
    
    def get_symbol_performance(self, symbol: str = None) -> List[Dict]:
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if symbol:
                cursor.execute('''
                    SELECT symbol, COUNT(*) as trades, SUM(quantity * price) as volume
                    FROM trades 
                    WHERE symbol = ?
                    GROUP BY symbol
                ''', (symbol,))
            else:
                cursor.execute('''
                    SELECT symbol, COUNT(*) as trades, SUM(quantity * price) as volume
                    FROM trades 
                    GROUP BY symbol
                    ORDER BY volume DESC
                ''')
            
            results = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "symbol": row[0],
                    "total_trades": row[1],
                    "total_volume": row[2]
                }
                for row in results
            ]
            
        except Exception as e:
            current_app.logger.error(f"종목별 성과 조회 실패: {e}")
            return []

trade_database = None

def get_trade_database() -> DatabaseService:
    global trade_database
    if trade_database is None:
        trade_database = DatabaseService()
    return trade_database