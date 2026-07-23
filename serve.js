// Minimal static file server used only by the test suite, so
// static-ngram-model.js's fetch() calls (which expect same-origin relative
// paths like "ngram-model/manifest.json") work the same way they would
// under GitHub Pages, without needing a browser.
const http = require('http');
const fs = require('fs');
const path = require('path');

const CONTENT_TYPES = {
  '.json': 'application/json',
  '.js': 'text/javascript',
  '.html': 'text/html',
  '.css': 'text/css',
  '.txt': 'text/plain',
};

function serve(rootDir, port) {
  return new Promise((resolve) => {
    const server = http.createServer((req, res) => {
      const filePath = path.join(rootDir, decodeURIComponent(req.url.split('?')[0]));
      fs.readFile(filePath, (err, data) => {
        if (err) { res.writeHead(404); res.end('not found'); return; }
        const ext = path.extname(filePath);
        res.writeHead(200, { 'Content-Type': CONTENT_TYPES[ext] || 'application/octet-stream' });
        res.end(data);
      });
    });
    server.listen(port, () => resolve(server));
  });
}

module.exports = { serve };
