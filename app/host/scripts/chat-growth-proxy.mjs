// Throwaway: serves the demo host unchanged, except POST /api/chat/stream, which streams a long scripted reply (NDJSON frames like the real host).
import http from 'node:http';
const UP = Number(process.env.UP ?? 18791), PORT = Number(process.env.PORT ?? 18790);
const TOK = Number(process.env.REPLY_TOKENS ?? 12000), REASON = Number(process.env.REASON_TOKENS ?? 3000), RATE = Number(process.env.RATE ?? 400);
const sleep = ms => new Promise(r => setTimeout(r, ms));
const para = i => `### Section ${i}\n\nThis is paragraph ${i} with **bold**, *italic*, \`inline code\` and a link [x](https://example.com/${i}). The measured value is about ${i} units and the results agree with the earlier runs, which is what we expected from the model.

More prose follows here so that a paragraph looks like real writing and not like a stress pattern.\n\n- item one of ${i}\n- item two of ${i}\n\n\`\`\`python\ndef f_${i}(a, b):\n    total = a + b * ${i}\n    return total\n\`\`\`\n\n| a | b |\n|---|---|\n| ${i} | ${i * 2} |\n\n`;
http.createServer(async (req, res) => {
  if (req.method === 'POST' && req.url.startsWith('/api/chat/stream')) {
    let raw = ''; for await (const c of req) raw += c;
    res.writeHead(200, { 'content-type': 'application/x-ndjson' });
    const w = o => res.write(JSON.stringify(o) + '\n');
    w({ type: 'start', model: 'Mock', contextMax: 131072, requestId: 'r' + Date.now() });
    let sent = 0, i = 0;
    const per = Math.max(1, Math.round(RATE / 50));
    while (sent < REASON) { w({ type: 'reasoning', text: 'thinking about step ' + (i++) + ' of the problem. '.repeat(1) }); sent += 9; if (sent % per < 9) await sleep(20); }
    sent = 0; i = 0;
    while (sent < TOK) { const p = para(i++); for (let k = 0; k < p.length; k += 24) { w({ type: 'delta', text: p.slice(k, k + 24) }); sent += 6; if ((sent / 6) % Math.max(1, Math.round(per / 2)) === 0) await sleep(10); } w({ type: 'stats', tokensOut: sent, tps: 90 }); }
    w({ type: 'done', tokensIn: 100, tokensOut: sent, tps: 90, durationMs: 1000, usageSource: 'engine' });
    return res.end();
  }
  const up = http.request({ host: '127.0.0.1', port: UP, path: req.url, method: req.method, headers: req.headers }, r => { res.writeHead(r.statusCode, r.headers); r.pipe(res); });
  up.on('error', () => { res.writeHead(502); res.end(); });
  req.pipe(up);
}).listen(PORT, '127.0.0.1', () => console.log('proxy on', PORT));
