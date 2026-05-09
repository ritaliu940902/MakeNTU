#include <Arduino.h>

// 定義控制參考電阻的數位腳位
const int pinR1 = 2;  // 控制 22Ω
const int pinR2 = 3;  // 控制 220Ω
const int pinR3 = 4;  // 控制 2.2kΩ
const int pinA0 = A0; // 量測節點

// 宣告參考電阻的精確阻值 (建議你可以用電表量過這三顆電阻後，把實際數值填進來會更準)
const float valR1 = 22.0;
const float valR2 = 220.0;
const float valR3 = 2200.0;

void setup() {
  Serial.begin(9600);
  
  // 初始時，將所有控制腳位設為 INPUT (高阻抗，相當於斷路)
  // 確保一開始沒有任何參考電阻干擾電路
  pinMode(pinR1, INPUT);
  pinMode(pinR2, INPUT);
  pinMode(pinR3, INPUT);
}

// 建立一個專門用來量測電阻的函式
float measureResistance(int activePin, float refValue) {
  // 1. 確保所有腳位都是斷開的
  pinMode(pinR1, INPUT);
  pinMode(pinR2, INPUT);
  pinMode(pinR3, INPUT);

  // 2. 啟動選定的參考電阻 (設為輸出並給予 5V 高電位)
  pinMode(activePin, OUTPUT);
  digitalWrite(activePin, HIGH);

  // 3. 稍微等待電壓穩定 (10 毫秒)
  delay(10);

  // 4. 讀取 A0 的 ADC 數值
  int adc = analogRead(pinA0);

  // 5. 量測完畢，立刻將該腳位切回高阻抗斷路狀態
  pinMode(activePin, INPUT);

  // 6. 防呆與邊界條件處理
  if (adc == 0) return 0.0;            // 測到 0 代表短路或 Rx 趨近於 0
  if (adc >= 1023) return -1.0;        // 測到 1023 代表開路 (沒接待測電阻)

  // 7. 帶入分壓公式計算未知電阻值
  // Rx = R1 * (ADC / (1023 - ADC))
  float rx = refValue * ((float)adc / (1023.0 - adc));
  return rx;
}

void loop() {
  float finalRx = 0.0;
  String usedResistor = "";

  // 【自動換檔邏輯】
  // 先用中間的 220Ω 來「試水溫」
  pinMode(pinR2, OUTPUT); 
  digitalWrite(pinR2, HIGH); 
  delay(10);
  int testADC = analogRead(pinA0);
  pinMode(pinR2, INPUT); // 試完立刻關閉

  // 根據試水溫的結果，決定真正要用哪一顆電阻來量
  // 理想的 ADC 讀值越靠近中間 (512) 越精準
  if (testADC > 850) {
      // 如果 ADC 很大，代表 Rx 比 220Ω 大很多，切換成 2.2kΩ 量測
      finalRx = measureResistance(pinR3, valR3);
      usedResistor = "2.2kΩ";
  } 
  else if (testADC < 150) {
      // 如果 ADC 很小，代表 Rx 比 220Ω 小很多，切換成 22Ω 量測
      finalRx = measureResistance(pinR1, valR1);
      usedResistor = "22Ω";
  } 
  else {
      // 落在中間合理範圍，直接用 220Ω 進行精確量測
      finalRx = measureResistance(pinR2, valR2);
      usedResistor = "220Ω";
  }

  // 測量完成，將結果印出來看
  Serial.println(finalRx);

  delay(500); // 每 0.5 秒更新一次數據
}