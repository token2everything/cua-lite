// Actual pinned Playwright classes; no browser/network in this unit harness.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const Module = require('node:module');
const path = require('node:path');
const [packageDir, overlayDir, scenario] = process.argv.slice(2);
const lib = path.join(packageDir, 'lib');
function load(relative) {
  const original = path.join(lib, 'server', relative);
  const mod = new Module(original, module);
  mod.filename = original;
  mod.paths = Module._nodeModulePaths(path.dirname(original));
  mod._compile(fs.readFileSync(path.join(overlayDir, relative), 'utf8'), original);
  return mod.exports;
}
const {HarTracer} = load('har/harTracer.js');
const {Tracing} = load('trace/recorder/tracing.js');
const {Response} = require(path.join(lib, 'server/network.js'));
const {Worker} = require(path.join(lib, 'server/page.js'));
const tick = () => new Promise(resolve => setImmediate(resolve));

function fixture(status = 302, finished = true) {
  const parent = {attribution: {}, instrumentation: {}};
  const worker = new Worker(parent, 'http://capture.invalid/worker.js');
  const request = {
    _context: parent, _responseEndTiming: -1, _failureText: null,
    _setResponse(response) { this.response = response; },
    frame: () => null, serviceWorker: () => worker,
    method: () => 'POST', url: () => 'http://capture.invalid/redirect',
    headers: () => [{name: 'Content-Type', value: 'application/x-www-form-urlencoded'}],
    rawRequestHeaders: () => Promise.resolve([]),
    headerValue: () => 'application/x-www-form-urlencoded',
    postDataBuffer: () => Buffer.from('product=30874'), bodySize: () => 13,
    redirectedFrom: () => null, _hasOverrides: () => false,
    resourceType: () => 'document',
  };
  const headers = [{name: 'Content-Type', value: 'text/html'},
    {name: 'Location', value: 'http://capture.invalid/wishlist'}];
  const timing = {startTime: Date.now(), domainLookupStart: -1, domainLookupEnd: -1,
    connectStart: -1, connectEnd: -1, secureConnectionStart: -1,
    requestStart: 0, responseStart: 1};
  const response = new Response(request, status, status === 302 ? 'Found' : 'OK',
    headers, timing, () => Promise.resolve(Buffer.from('body')), false, 'HTTP/1.1');
  response._serverAddrFinished();
  response._securityDetailsFinished();
  if (finished) response._requestFinished(2);
  return {request, response, worker, release() {
    response.setRawResponseHeaders([...headers, {name: 'Set-Cookie', value: 'capture_probe=complete; Path=/'}]);
  }};
}

function setup(first) {
  const chunks = [];
  let exported;
  const tracing = Object.create(Tracing.prototype);
  tracing._allResources = new Set();
  tracing._pendingHarEntries = new Set();
  tracing._eventListeners = [];
  tracing._isStopping = false;
  tracing._state = {recording: true, options: {}, networkSha1s: new Set(),
    traceSha1s: new Set(), networkFile: 'memory.network', traceFile: 'memory.trace',
    tracesDir: 'memory', traceName: 'test', chunkOrdinal: 1, resourcesDir: 'resources'};
  tracing._context = {instrumentation: {removeListener() {}}};
  tracing._fs = {appendFile(file, data) { chunks.push(data); },
    copyFile() { exported = chunks.join(''); },
    async syncAndGetError() { return undefined; }, writeFile() {}};
  const tracer = new HarTracer({_options: {}}, null, tracing, {
    content: 'attach', includeTraceInfo: true, recordRequestOverrides: false,
    waitForContentOnStop: false, omitSizes: true,
  });
  tracing._harTracer = tracer;
  tracer._started = true;
  const add = entry => { tracer._onRequest(entry.request); tracer._onResponse(entry.response); };
  add(first);
  return {tracing, tracer, add, exported: () => exported,
    rows: () => chunks.join('').trim().split('\n').filter(Boolean).map(JSON.parse)};
}

(async () => {
  const first = fixture(scenario === 'normal200' || scenario === 'stream' ? 200 : 302, scenario !== 'stream');
  const run = setup(first);
  if (scenario === 'failed') {
    first.request._failureText = 'net::ERR_ABORTED';
    run.tracer._onRequestAborted(first.request);
    await run.tracer._onRequestFailed(first.request);
    assert.equal(run.rows()[0].snapshot.response._failureText, 'net::ERR_ABORTED');
    assert.equal(run.rows()[0].snapshot._wasAborted, true);
  } else if (scenario === 'stream') {
    assert.equal(run.tracer._pendingFinishedEntries.size, 0);
    await run.tracing.stopChunk({mode: 'entries'});
    assert.equal(run.rows().length, 1);
  } else {
    await run.tracer._onRequestFinished(first.request, first.response);
    await tick();
    if (scenario === 'closing' || scenario === 'deadline') {
      // Shorten only this unit clock, not the shipped source or live deadline.
      if (scenario === 'deadline') {
        const nativeTimeout = global.setTimeout;
        global.setTimeout = (fn, ms, ...args) => nativeTimeout(fn, ms === 30000 ? 20 : ms, ...args);
      }
      const stopped = run.tracing.stopChunk({mode: 'entries'});
      await tick();
      if (scenario === 'closing') first.worker.didClose();
      await assert.rejects(stopped, scenario === 'closing' ? /metadata capture incomplete/ : /metadata capture timeout/);
      assert.equal(run.exported(), undefined);
      assert.equal(run.tracer._started, false);
      assert.equal(run.tracing._isStopping, true);
      await assert.rejects(run.tracing.stopChunk({mode: 'entries'}), /already stopping/);
      first.release();
      await tick();
      assert.equal(run.rows().length, 0);
    } else {
      let done = false;
      const stopped = run.tracing.stopChunk({mode: 'entries'}).then(() => { done = true; });
      await tick();
      if (scenario === 'concurrent') {
        const second = fixture();
        run.add(second);
        await run.tracer._onRequestFinished(second.request, second.response);
        first.release();
        await tick();
        assert.equal(done, false);
        second.release();
      } else first.release();
      await stopped;
      const rows = run.rows();
      assert.equal(rows.length, scenario === 'concurrent' ? 2 : 1);
      for (const row of rows) assert.equal(row.snapshot.response.cookies[0]?.name, 'capture_probe');
      assert.equal(run.exported(), rows.map(row => JSON.stringify(row)).join('\n') + '\n');
      if (scenario === 'normal200') {
        assert.ok(rows[0].snapshot.request.postData._sha1);
        assert.ok(rows[0].snapshot.response.content._sha1);
      }
    }
  }
  console.log(JSON.stringify({scenario, passed: true, realClasses: true, modelCalls: 0, network: false}));
})().catch(error => { console.error(error); process.exitCode = 1; });
