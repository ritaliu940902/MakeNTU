from machine import ADC, Pin
import time

# 1. 初始化 ADC (感應點)
adc = ADC(Pin(26))

# 2. 定義參考電阻腳位與數值 (依照你的設定)
ref_configs = {
    "low":  {"pin": Pin(13, Pin.IN), "val": 1000},  # 1k
    "mid":  {"pin": Pin(14, Pin.IN), "val": 9090},  # 9.09k
    "high": {"pin": Pin(15, Pin.IN), "val": 90900}  # 90.9k
}

def set_range(mode):
    """ 切換 GPIO 狀態：只讓目標腳位輸出 3.3V，其餘維持高阻抗(IN) """
    for m in ref_configs:
        ref_configs[m]["pin"].init(Pin.IN)
    
    target = ref_configs[mode]["pin"]
    target.init(Pin.OUT)
    target.value(1) # 提供 3.3V 電源
    return ref_configs[mode]["val"]

def get_smooth_raw(samples=100):
    """ 連續取樣取平均，減少雜訊 """
    total = 0
    for _ in range(samples):
        total += adc.read_u16()
    return total / samples

# 啟動初始檔位 (先從中間檔開始)
current_mode = "mid"
r_ref = set_range(current_mode)

print("--- MakeNTU 2026: 高精度自動量測系統啟動 ---")

while True:
    raw = get_smooth_raw()
    
    # 3. 自動換檔邏輯 (設定 15% ~ 85% 的有效量測區間)
    changed = False
    
    if raw < 10000: # 讀值太小，代表 Rx 相對參考電阻太小 -> 降檔
        if current_mode == "high":
            current_mode = "mid"
            changed = True
        elif current_mode == "mid":
            current_mode = "low"
            changed = True
            
    elif raw > 55000: # 讀值太大，代表 Rx 相對參考電阻太大 -> 升檔
        if current_mode == "low":
            current_mode = "mid"
            changed = True
        elif current_mode == "mid":
            current_mode = "high"
            changed = True

    if changed:
        r_ref = set_range(current_mode)
        time.sleep(0.1) # 給電路一點點時間穩定電荷
        continue # 跳過本次計算，直接重新量測

    # 4. 計算阻值 (分壓公式)
    if 500 < raw < 65000:
        # Rx = R_ref * (Vout / (Vin - Vout))
        rx = r_ref * (raw / (65535 - raw))
        
        # 漂亮的單位格式化輸出
        if rx >= 1000:
            print(f"[{current_mode:^4}] 阻值: {rx/1000:8.3f} kΩ (Raw: {int(raw)})")
        else:
            print(f"[{current_mode:^4}] 阻值: {rx:8.1f} Ω  (Raw: {int(raw)})")
    else:
        print(f"[{current_mode:^4}] 等待電阻接入或超出量程...")

    time.sleep(0.5)