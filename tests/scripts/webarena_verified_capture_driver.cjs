// Live-test bootstrap: reorder one real CDP ExtraInfo event in an isolated driver.
const path = require('node:path');
const fs = require('node:fs');
const packageDir = process.env.TEST_CAPTURE_PACKAGE;
const mode = process.env.TEST_CAPTURE_MODE;
const {CRNetworkManager} = require(path.join(packageDir, 'lib/server/chromium/crNetworkManager.js'));
const receive = CRNetworkManager.prototype._onResponseReceivedExtraInfo;
const heldByManager = new WeakMap();
CRNetworkManager.prototype._onResponseReceivedExtraInfo = function(event) {
  let held = heldByManager.get(this);
  if (!held) {
    held = new Map();
    heldByManager.set(this, held);
  }
  const queued = held.get(event.requestId);
  if (queued) {
    queued.push(event);
    fs.appendFileSync(process.env.TEST_CAPTURE_EVENTS, JSON.stringify({type: 'same_chain_extra_info_queued', requestId: event.requestId, statusCode: event.statusCode}) + '\n');
    return;
  }
  const cookie = Object.entries(event.headers).find(([name]) => name.toLowerCase() === 'set-cookie')?.[1] || '';
  if (cookie.startsWith('capture_probe=')) {
    fs.appendFileSync(process.env.TEST_CAPTURE_EVENTS, JSON.stringify({type: 'real_extra_info_intercepted', requestId: event.requestId, statusCode: event.statusCode, mode, at: Date.now()}) + '\n');
    const pending = [event];
    held.set(event.requestId, pending);
    if (mode === 'delayed') {
      // A redirect chain reuses its request ID. Preserve ExtraInfo FIFO within
      // that chain, so this delay does not misattribute its next response.
      setTimeout(() => {
        held.delete(event.requestId);
        for (const responseEvent of pending) {
          fs.appendFileSync(process.env.TEST_CAPTURE_EVENTS, JSON.stringify({type: 'same_chain_extra_info_released', requestId: responseEvent.requestId, statusCode: responseEvent.statusCode}) + '\n');
          receive.call(this, responseEvent);
        }
      }, 250);
    }
    return;
  }
  return receive.call(this, event);
};
require(path.join(packageDir, 'cli.js'));
