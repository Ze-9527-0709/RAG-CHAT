// Lightweight crypto.getRandomValues polyfill for build environments lacking Web Crypto.
// Not cryptographically secure; only for non-security IDs.
if (typeof globalThis !== 'undefined' && (!(globalThis as any).crypto || !(globalThis as any).crypto.getRandomValues)) {
  (globalThis as any).crypto = (globalThis as any).crypto || {};
  (globalThis as any).crypto.getRandomValues = function<T extends ArrayBufferView>(arr: T): T {
    if (!(arr && arr.buffer instanceof ArrayBuffer)) throw new TypeError('Expected an ArrayBufferView');
    for (let i = 0; i < (arr as any).length; i++) {
      (arr as any)[i] = Math.floor(Math.random()*256);
    }
    return arr;
  };
}
