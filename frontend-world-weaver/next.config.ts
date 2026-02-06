import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // basePath 始终启用
  basePath: '/tools/world-weaver',
  // 输出为 standalone 模式，适合 Docker 部署
  output: 'standalone',
};

export default nextConfig;
