/** @type {import('next').NextConfig} */
const nextConfig = {
  // `standalone` empaqueta solo lo necesario para ejecutar: la imagen final
  // pesa una fracción de lo que ocuparía node_modules completo.
  output: "standalone",
  reactStrictMode: true,
  poweredByHeader: false,
};

export default nextConfig;
