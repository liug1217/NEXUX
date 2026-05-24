// api/chat.js
// 這是專門跑在 Vercel 雲端的 AI 大腦程式碼

export default async function handler(req, res) {
  // 1. 防止跨網域錯誤（CORS），確保前端網頁能順利連線
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // 2. 檢查前端傳過來的提問
  if (req.method !== 'POST') {
    return res.status(405).json({ error: '請使用 POST 方法傳送訊息' });
  }

  try {
    const { message } = req.body; // 抓取使用者輸入的對話

    // 3. 呼叫 OpenAI / ChatGPT 的官方雲端接口
    // Vercel 會自動安全地去抓取儲存在後台的 API Key (process.env.OPENAI_API_KEY)
    const response = await fetch('https://openai.com', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`
      },
      body: JSON.stringify({
        model: 'gpt-4o-mini', // 使用高智商且便宜好用的模型
        messages: [{ role: 'user', content: message }],
        temperature: 0.7
      })
    });

    const data = await response.json();

    // 4. 把 AI 的回答結果乾乾淨淨地吐回給前端網頁
    if (data.choices && data.choices[0]) {
      const aiReply = data.choices[0].message.content;
      return res.status(200).json({ reply: aiReply });
    } else {
      return res.status(500).json({ error: 'AI 服務器回應異常' });
    }

  } catch (error) {
    // 萬一出錯，雲端會捕捉並報錯，網頁不會死機
    return res.status(500).json({ error: error.message });
  }
}
