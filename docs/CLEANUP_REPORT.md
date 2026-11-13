# 語料整理報告(data/ 大掃除)

> 日期:2026-11-13
> 動作:把 `data/` 碎片化的語料重新整理、去重、分類。
> **只重新分檔 + 去除完全重複的對話,沒有改動任何一句對話內容。**

## 一、整理前的狀況(「亂」在哪)

| 問題 | 數量 |
|------|------|
| 檔案過度碎片化 | **487** 個 `.jsonl`(光 `taiwan_*` 就 103 個、`quick_chitchat_*` 22 個、`multiturn_*` 16 個、`science_*` 12 個、`cooking_*` 11 個…) |
| 完全重複的對話 | 86 組、**166 筆多餘**(例:`identity_reinforced.jsonl` 整份內容被複製了 3 次) |
| 同一問句重複出現 | 467 組、853 筆(「你好」出現 48 次,答案還各不相同) |
| 身分回答不一致 | 詳見第三節 |

好消息:**格式本身是乾淨的**——0 個 JSON 壞行、0 個 role 結構錯誤、0 個空檔案。

## 二、整理後的結果

- **487 檔 → 28 個分類檔**
- **10,055 筆 → 9,889 筆**(移除 166 筆完全重複;驗證:9889 + 166 = 10055 ✅,沒掉任何獨特對話)
- 重跑稽核:0 壞檔、0 格式問題、**完全重複已歸零**

### 新的分類檔一覽

| 檔名 | 筆數 | 內容 |
|------|-----:|------|
| `01_identity.jsonl` | 38 | NEXUX 自我認同、關於 AI 本身 |
| `02_greetings_smalltalk.jsonl` | 2933 | 問候、閒聊、quick_chitchat、短對話 |
| `03_multiturn.jsonl` | 103 | 多輪對話 |
| `04_daily_life.jsonl` | 123 | 日常生活、常識 |
| `10_food_cooking.jsonl` | 125 | 飲食、烹飪、食譜(非台灣專屬) |
| `11_mental_health.jsonl` | 174 | 心理健康、情緒、心理學 |
| `12_health_wellness.jsonl` | 105 | 身體健康、養生、醫療、銀髮 |
| `13_career_work.jsonl` | 93 | 職涯、工作、職場 |
| `14_relationships_family.jsonl` | 121 | 感情、家庭、育兒、婚姻、青少年 |
| `15_science.jsonl` | 385 | 科學、生物、化學、物理、地科、地震 |
| `16_tech_ai.jsonl` | 182 | 科技、AI、電腦、網路、數位生活 |
| `17_travel.jsonl` | 49 | 旅遊(非台灣) |
| `18_education_language.jsonl` | 214 | 教育、語言學習、英語、成語、校園 |
| `19_arts_creative.jsonl` | 166 | 藝術、創意寫作、音樂、興趣、詩詞、故事 |
| `20_taiwan_food.jsonl` | 59 | 台灣飲食、夜市、小吃、咖啡 |
| `21_taiwan_travel_geo.jsonl` | 70 | 台灣旅遊、地理、景點、交通、自然 |
| `22_taiwan_culture_society.jsonl` | 302 | 台灣文化、歷史、政治、社會、教育、宗教、原民、語言 |
| `23_taiwan_life_misc.jsonl` | 123 | 台灣其他生活(居住、職場、財經、運動、遊戲…) |
| `24_finance_money.jsonl` | 81 | 理財、金錢、投資、經濟、創業 |
| `25_philosophy_growth.jsonl` | 136 | 哲學、自我成長、宗教心靈 |
| `26_nature_environment.jsonl` | 128 | 自然、環境、動物、寵物、天氣 |
| `27_society_community.jsonl` | 94 | 社群、社會議題、志工、溝通、文化 |
| `28_world_knowledge.jsonl` | 2329 | 世界/一般知識、QA、延伸問答、時事、趣味知識 |
| `29_sports_entertainment.jsonl` | 96 | 運動、娛樂、遊戲 |
| `30_life_advice.jsonl` | 158 | 生活建議、技巧、實用技能 |
| `31_coding_tech.jsonl` | 1130 | 程式(Python/JS/HTML/CSS)、資安、網路、Minecraft |
| `32_math_logic.jsonl` | 294 | 數學、邏輯、週期表 |
| `33_home_lifestyle.jsonl` | 73 | 居家、DIY、購物、時尚、節慶 |

> 說明:訓練/推論的程式(`messages_format.py` 用 `glob("*.jsonl")`)是讀 `data/` 底下**所有** `.jsonl`,跟檔名無關,所以重新分檔不影響任何既有流程。

## 三、身分語料(你最早的顧慮)

查了所有被問到身分的對話:

- ✅ **27 筆正確自稱 NEXUX**
- ✅ **沒有任何一筆錯說自己是 GPT / ChatGPT**
  - 有一筆講「我是基於 GPT2 架構、在台灣資料上微調的 NEXUX 語言模型」——這是**正確且誠實**的(架構確實來自 GPT-2,但明確說自己叫 NEXUX)。
- ⚠️ **6 筆答得太籠統**(被問身分時只說「我是AI助手 / 語言模型」,沒提到 NEXUX)。這不是錯誤,是風格一致性問題,**我沒有自動改**,留給你決定:

| 位置 | 問 | 答 |
|------|----|----|
| `02_greetings_smalltalk.jsonl:188` | 你叫什麼? | 我是AI助手。我可以回答問題。 |
| `02_greetings_smalltalk.jsonl:638` | 你是什麼東西? | 我是AI,也就是人工智慧,不是實體的東西…… |
| `02_greetings_smalltalk.jsonl:653` | 你是AI嗎 | 是的,我是AI——一個語言模型…… |
| `02_greetings_smalltalk.jsonl:1280` | 你是AI嗎? | 是的,我是AI助手…… |
| `02_greetings_smalltalk.jsonl:1381` | 你是真人嗎 | 是的,我是AI助手…… |
| `02_greetings_smalltalk.jsonl:1500` | 你是AI嗎 | 是的,我是一個AI助手…… |

**建議**:如果希望 NEXUX 被問身分時一律講出自己的名字,可以把這 6 筆的答案補上「我是 NEXUX」,或在 `01_identity.jsonl` 多加幾筆標準自我介紹來強化。這部分等你回來再一起決定。

## 四、還沒處理、留給你決定的

1. **同問句不同答(408 組、687 筆)**:例如「你好」有 46 種不同回答。對聊天機器人來說這是「回答多樣性」,不一定要刪;但若有互相矛盾的,值得挑掉。**我沒有自動刪。**
2. **上述 6 筆身分語料的措辭。**
3. **要不要重新訓練 + 匯出 + push**:這次整理移除了 166 筆重複(約 1.6%),對模型影響很小。重訓需要 torch/GPU、耗時,且會觸發正式站重新部署,所以**我沒有自動執行**,等你回來決定。

## 五、怎麼還原(萬一不滿意)

所有變更都還在 git。要回到整理前:

```bash
git checkout -- data/
```

(只要還沒 commit,這行就能把 `data/` 還原成原本的 487 個檔。)
