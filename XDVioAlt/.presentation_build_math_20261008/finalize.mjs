import path from 'node:path';
import { pathToFileURL } from 'node:url';
const root = '/home/aarf101/projtutore';
const skillDir = '/home/aarf101/.codex/plugins/cache/openai-primary-runtime/presentations/26.915.20218/skills/presentations';
const buildDir = path.join(root, '.presentation_build_math_20261008');
const { finalizePresentation } = await import(pathToFileURL(path.join(skillDir, 'container_tools/artifact_tool_utils.mjs')).href);
const result = await finalizePresentation({
  explicitTotalSlideCount: 7,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [7],
  materializeLiteralChartWorkbooks: true,
  workspaceDir: root,
  candidatePath: path.join(buildDir, 'safewatch_math_draft.pptx'),
  finalPath: path.join(root, 'presentations', 'SafeWatch_Model_Math_20261008_v2.pptx'),
  pythonExecutable: '/home/aarf101/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3.12',
  integrityValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(skillDir, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-bullet-geometry', '--validate-heading-fit'],
  fontPolicy: { basis: 'design', families: ['DejaVu Sans'] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(buildDir, 'SafeWatch_Model_Math_20261008_v2.validation.json'),
});
console.log(JSON.stringify(result, null, 2));
