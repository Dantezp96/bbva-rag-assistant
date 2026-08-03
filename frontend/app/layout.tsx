import type { Metadata } from "next";
import { IBM_Plex_Mono, Public_Sans, Source_Serif_4 } from "next/font/google";

import "./globals.css";
import { Shell } from "@/components/Shell";

// Sustitutos libres de las familias propietarias de BBVA:
//   Benton Sans BBVA  ->  Public Sans   (mismo esqueleto News Gothic)
//   Tiempos Headline  ->  Source Serif  (serif transicional de alto contraste)
// El monoespaciado se reserva para cifras y telemetría, que aquí son contenido.
const sans = Public_Sans({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
  variable: "--font-public-sans",
  display: "swap",
});
const serif = Source_Serif_4({
  subsets: ["latin"],
  weight: ["600", "700"],
  variable: "--font-source-serif",
  display: "swap",
});
const mono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-plex-mono",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Asistente RAG · BBVA Colombia",
  description:
    "Asistente conversacional sobre el contenido publicado en bbva.com.co, con recuperación aumentada y citación de fuentes.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="es"
      className={`${sans.variable} ${serif.variable} ${mono.variable}`}
    >
      <body>
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
