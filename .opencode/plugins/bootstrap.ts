// Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
//
// 项目级 opencode 插件：在 opencode 启动时自动调用 .agents/setup_agent.sh
// 完成 .opencode/skills 等软链初始化，免去手工执行 setup_agent.sh 的步骤。


import type { Plugin } from "@opencode-ai/plugin"
import * as fs from "fs"
import * as path from "path"

function findGitRoot(startDir: string): string {
  let dir = startDir
  while (dir !== path.dirname(dir)) {
    if (fs.existsSync(path.join(dir, ".git"))) return dir
    dir = path.dirname(dir)
  }
  return startDir
}

function hasBash(): boolean {
  try {
    require("child_process").execSync("bash --version", { stdio: "ignore" })
    return true
  } catch {
    return false
  }
}

export const BootstrapPlugin: Plugin = async ({ $, directory }) => {
  const rootDir = findGitRoot(directory)
  const setupScript = path.join(rootDir, ".agents", "setup_agent.sh")
  const logFile = path.join(rootDir, ".opencode", "bootstrap.log")

  const run = async () => {
    if (!fs.existsSync(setupScript)) {
      // 没有 .agents/setup_agent.sh 就直接放行，不阻塞 opencode 启动
      return
    }

    if (!hasBash()) {
      process.stdout.write(
        `💡 bash is missing in the current environment, please manually run ${setupScript} --agent opencode\n\n`,
      )
      return
    }

    try {
      // --quiet：脚本本身不打印过程；插件再用 setTimeout 给一次必要提示
      await $`bash ${setupScript} --agent opencode --quiet`
    } catch (error: any) {
      const detail = error?.stderr
        ? `${error.message}\n${error.stderr}`
        : error?.message ?? String(error)
      try {
        fs.appendFileSync(
          logFile,
          `[${new Date().toISOString()}] bootstrap failed: ${detail}\n`,
        )
      } catch {}
      setTimeout(() => {
        process.stdout.write(
          `❌ .agents/setup_agent.sh failed, see ${logFile}\n`,
        )
      }, 1500)
    }
  }

  // 立即触发，不阻塞插件返回
  run()

  return {
    event: async () => {},
  }
}
