import { spawn, spawnSync } from 'node:child_process';
import { closeSync, existsSync, mkdirSync, openSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const moduleFile = fileURLToPath(import.meta.url);
const root = path.resolve(path.dirname(moduleFile), '..', '..');
const url = 'http://127.0.0.1:8194';
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function json(route, init) {
  const response = await fetch(url + route, init);
  const body = await response.json();
  if (!response.ok) throw new Error(body.error?.message || body.error || `${route}: ${response.status}`);
  return body;
}

async function main() {
  const adapterPath = process.env.CUBQL_HOST_ADAPTER;
  const stableRuntime = process.env.CUBQL_STABLE_RUNTIME;
  if (!adapterPath) throw new Error('Set CUBQL_HOST_ADAPTER to the host integration module.');
  if (!stableRuntime) throw new Error('Set CUBQL_STABLE_RUNTIME to the native runtime root.');
  if (!process.env.AMD_UV_INPUT || !process.env.AMD_UV_GLB) {
    throw new Error('Set AMD_UV_INPUT and AMD_UV_GLB to the capture inputs.');
  }
  const { createCaptureLaunch, buildUvAtlasWorkflow } = await import(pathToFileURL(path.resolve(adapterPath)).href);
  if (typeof createCaptureLaunch !== 'function' || typeof buildUvAtlasWorkflow !== 'function') {
    throw new Error('The host adapter must export createCaptureLaunch and buildUvAtlasWorkflow.');
  }

  const diagnostics = path.join(root, 'data', 'runtime', 'diagnostics', 'native-amd');
  const captureFile = path.join(diagnostics, 'baton-cubql-capture.pt');
  const reportFile = path.join(diagnostics, 'baton-cubql-capture.json');
  const inputImage = path.resolve(process.env.AMD_UV_INPUT);
  const inputGlb = path.resolve(process.env.AMD_UV_GLB);
  if (!existsSync(inputImage)) throw new Error(`Input PNG not found: ${inputImage}`);
  if (!existsSync(inputGlb)) throw new Error(`Input GLB not found: ${inputGlb}`);
  if (await fetch(url + '/system_stats', { signal: AbortSignal.timeout(1500) }).then((r) => r.ok).catch(() => false)) {
    throw new Error('Port 8194 is already answering; refusing to reuse or stop another process.');
  }

  mkdirSync(diagnostics, { recursive: true });
  rmSync(captureFile, { force: true });
  rmSync(reportFile, { force: true });
  const env = {
    ...process.env,
    CUBQL_REPO: root,
    CUBQL_STABLE_RUNTIME: path.resolve(stableRuntime),
    CUBQL_EXPERIMENTAL_RUNTIME: path.join(root, 'data', 'runtime', 'experimental-runtime'),
    CUBQL_CAPTURE: captureFile,
    AMD_UV_INPUT: inputImage,
    AMD_UV_GLB: inputGlb,
  };
  const baseLaunch = await createCaptureLaunch({
    root,
    diagnostics,
    captureFile,
    url,
    env,
    tempDir: path.join(diagnostics, 'cubql-capture-temp'),
    dbFile: path.join(diagnostics, 'cubql-capture.db'),
  });
  const launch = {
    ...baseLaunch,
    args: ['-I', path.join(root, 'tools', 'acceptance', 'comfy_capture.py'), ...baseLaunch.args.slice(3)],
  };
  const logFile = path.join(diagnostics, 'baton-cubql-capture-server.log');
  const fd = openSync(logFile, 'w');
  const child = spawn(launch.command, launch.args, {
    cwd: launch.cwd, env: launch.env, windowsHide: true, stdio: ['ignore', fd, fd],
  });
  closeSync(fd);
  let vramStarted = false;
  const deadline = Date.now() + 275000;
  const started = Date.now();
  try {
    let ready = false;
    while (Date.now() < deadline && !ready) {
      if (child.exitCode !== null) throw new Error(`Capture ComfyUI exited during startup (${child.exitCode}). See ${logFile}`);
      ready = await fetch(url + '/system_stats', { signal: AbortSignal.timeout(1500) }).then((r) => r.ok).catch(() => false);
      if (!ready) await sleep(1000);
    }
    if (!ready) throw new Error(`Capture ComfyUI startup timed out. See ${logFile}`);

    const form = new FormData();
    form.append('image', new Blob([readFileSync(inputImage)], { type: 'image/png' }), 'baton-cubql-capture.png');
    form.append('overwrite', 'true');
    const uploaded = await json('/upload/image', { method: 'POST', body: form });
    const workflow = buildUvAtlasWorkflow({ imageName: uploaded.name, glbPath: inputGlb, textureSize: 2048 });
    vramStarted = await fetch(url + '/af/vram/start', { method: 'POST', signal: AbortSignal.timeout(3000) }).then((r) => r.ok).catch(() => false);
    const queued = await json('/prompt', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ prompt: workflow }),
    });
    while (Date.now() < deadline && !existsSync(captureFile)) {
      if (child.exitCode !== null) throw new Error(`Capture ComfyUI exited during generation (${child.exitCode}). See ${logFile}`);
      const history = await json('/history/' + queued.prompt_id);
      const record = history[queued.prompt_id];
      const error = record?.status?.messages?.find((message) => message[0] === 'execution_error');
      if (error && !existsSync(captureFile)) throw new Error(error[1]?.exception_message || 'Capture generation failed.');
      if (!existsSync(captureFile)) await sleep(1500);
    }
    if (!existsSync(captureFile)) throw new Error(`Capture generation timed out. See ${logFile}`);
    const vram = vramStarted ? await json('/af/vram/stop', { method: 'POST' }).catch(() => null) : null;
    vramStarted = false;
    const report = {
      stage: 'capture-before-postprocess',
      complete: true,
      seconds: Math.round((Date.now() - started) / 100) / 10,
      captureFile,
      captureBytes: readFileSync(captureFile).length,
      inputImage,
      inputGlb,
      textureSize: 2048,
      seed: 42,
      vram,
      fallbackEvents: [],
    };
    writeFileSync(reportFile, JSON.stringify(report, null, 2) + '\n');
    console.log(JSON.stringify(report, null, 2));
  } finally {
    if (vramStarted) await fetch(url + '/af/vram/stop', { method: 'POST' }).catch(() => {});
    await fetch(url + '/free', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ unload_models: true, free_memory: true }),
    }).catch(() => {});
    if (child.exitCode === null && process.platform === 'win32') {
      spawnSync('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
    }
  }
}

await main();
