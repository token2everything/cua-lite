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
  const context = {_options: {}, instrumentation: {removeListener() {}},
    attribution: {playwright: {options: {sdkLanguage: 'javascript'}}}};
  const tracing = new Tracing(context, 'memory');
  tracing._state = {recording: true, options: {}, networkSha1s: new Set(),
    traceSha1s: new Set(), networkFile: 'memory.network', traceFile: 'memory.trace',
    tracesDir: 'memory', traceName: 'test', chunkOrdinal: 1, resourcesDir: 'resources'};
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
  if (first) add(first);
  return {tracing, tracer, add, exported: () => exported,
    rows: () => chunks.join('').trim().split('\n').filter(Boolean).map(JSON.parse)};
}

function orderEntry(id, time, status) {
  return {_probe_id: id, startedDateTime: time,
    request: {method: 'POST', url: 'http://capture.invalid/-/profile',
      headers: [], cookies: [], postData: {mimeType: 'application/x-www-form-urlencoded', text: 'name=probe'}},
    response: {status, headers: [], cookies: [], content: {mimeType: 'text/html'}}};
}

async function checkOrder() {
  const run = setup();
  const firstTime = '2026-10-06T05:58:29.696Z';
  const lastTime = '2026-10-06T05:58:55.814Z';
  const ids = () => run.rows().map(row => row.snapshot._probe_id);
  if (scenario === 'order451' || scenario === 'orderPair' || scenario === 'orderTriple') {
    const entries = scenario === 'order451'
      ? [orderEntry('har73', firstTime, 200), orderEntry('har74', firstTime, 302),
        orderEntry('har76', lastTime, 200), orderEntry('har77', lastTime, 302)]
      : scenario === 'orderTriple'
      ? [orderEntry('first200', lastTime, 200), orderEntry('second201', lastTime, 201),
        orderEntry('third302', lastTime, 302)]
      : [orderEntry('slow200', lastTime, 200), orderEntry('fast302', lastTime, 302)];
    for (const entry of entries) run.tracing.onEntryStarted(entry);
    const completions = scenario === 'order451' ? [0, 1, 3, 2]
      : entries.map((_, index) => index).reverse();
    for (const index of completions) run.tracing.onEntryFinished(entries[index]);
    assert.deepEqual(ids(), entries.map(entry => entry._probe_id));
    assert.equal(run.rows().at(-1).snapshot.response.status, 302);
    assert.equal(run.tracing._pendingHarEntries.size, 0);
  } else if (scenario === 'orderStream') {
    const stream = orderEntry('old-stream', firstTime, -1);
    const ready = orderEntry('new-ready', lastTime, 200);
    run.tracing.onEntryStarted(stream);
    run.tracing.onEntryStarted(ready);
    run.tracing.onEntryFinished(ready);
    assert.deepEqual(ids(), ['new-ready']);
    await run.tracing.stopChunk({mode: 'entries'});
    assert.deepEqual(ids(), ['new-ready', 'old-stream']);
    assert.equal(run.rows()[1].snapshot.response.status, -1);
  } else if (scenario === 'orderSameStream') {
    const stream = fixture(200, false);
    const ready = fixture(200);
    run.add(stream);
    run.add(ready);
    for (const item of [stream, ready])
      run.tracer._entryForRequest(item.request).startedDateTime = lastTime;
    await run.tracer._onRequestFinished(ready.request, ready.response);
    ready.release();
    await tick();
    assert.equal(run.rows().length, 0);
    assert.equal(run.tracer._pendingFinishedEntries.size, 0);
    assert.equal(run.tracing._pendingHarEntries.size, 2);
    await run.tracing.stopChunk({mode: 'entries'});
    const [partial, finished] = run.rows().map(row => row.snapshot);
    assert.deepEqual(run.rows().map(row => row.snapshot.response.status), [200, 200]);
    assert.equal(partial.response.content._sha1, undefined);
    assert.ok(finished.response.content._sha1);
    assert.equal(finished.response.cookies[0].name, 'capture_probe');
    assert.equal(run.tracing._pendingHarEntries.size, 0);
  } else if (scenario === 'orderFailed') {
    const aborted = fixture();
    const ready = fixture();
    run.add(aborted);
    run.add(ready);
    for (const item of [aborted, ready])
      run.tracer._entryForRequest(item.request).startedDateTime = lastTime;
    await run.tracer._onRequestFinished(ready.request, ready.response);
    ready.release();
    await tick();
    assert.equal(run.rows().length, 0);
    aborted.request._failureText = 'net::ERR_ABORTED';
    run.tracer._onRequestAborted(aborted.request);
    await run.tracer._onRequestFailed(aborted.request);
    assert.equal(run.rows().length, 2);
    assert.equal(run.rows()[0].snapshot.response._failureText, 'net::ERR_ABORTED');
    assert.equal(run.rows()[0].snapshot._wasAborted, true);
    assert.equal(run.rows()[1].snapshot.response.cookies[0].name, 'capture_probe');
  } else if (scenario === 'orderLateFull') {
    const stream = orderEntry('stream', lastTime, -1);
    run.tracing.onEntryStarted(stream);
    await run.tracing.stopChunk({mode: 'entries'});
    stream.response.status = 200;
    run.tracing.onEntryFinished(stream);
    // The baseline already writes partial+full entries across chunks.
    assert.deepEqual(ids(), ['stream', 'stream']);
    assert.deepEqual(run.rows().map(row => row.snapshot.response.status), [-1, 200]);
  } else if (scenario === 'orderAcrossChunk') {
    const slow = orderEntry('slow200', lastTime, -1);
    const fast = orderEntry('fast302', lastTime, 302);
    run.tracing.onEntryStarted(slow);
    run.tracing.onEntryStarted(fast);
    run.tracing.onEntryFinished(fast);
    await run.tracing.stopChunk({mode: 'entries'});
    slow.response.status = 200;
    run.tracing.onEntryFinished(slow);
    assert.equal(run.rows().length, 3);
    assert.equal(ids().filter(id => id === 'slow200').length, 2);
    assert.equal(ids().at(-1), 'slow200');
    assert.equal(run.rows().at(-1).snapshot.response.status, 200);
    // This repair does not remove inherited late-full ordering drift across chunks.
  }
  return {scenario, passed: true, realClasses: true, modelCalls: 0, network: false,
    ids: ids(), statuses: run.rows().map(row => row.snapshot.response.status)};
}

(async () => {
  if (scenario.startsWith('order')) {
    console.log(JSON.stringify(await checkOrder()));
    return;
  }
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
