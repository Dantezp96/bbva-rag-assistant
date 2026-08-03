"use client";

import { Fragment, type ReactNode } from "react";

/**
 * Render mínimo del Markdown que devuelve el modelo.
 *
 * Se hace a mano en lugar de traer una librería por dos razones: el modelo
 * produce un subconjunto muy acotado (párrafos, viñetas, negritas, `código`)
 * y, sobre todo, **no se inyecta HTML**. El texto viene de un LLM; convertirlo
 * en marcado arbitrario sería abrir la puerta a que un contenido malicioso
 * indexado acabe ejecutándose en el navegador.
 *
 * Los marcadores de cita `[n]` se resaltan para unir visualmente la frase con
 * su fuente en el panel de abajo.
 */
function inline(texto: string, clave: string): ReactNode[] {
  const partes: ReactNode[] = [];
  const patron = /(\*\*[^*]+\*\*|`[^`]+`|\[\d{1,2}\])/g;
  let ultimo = 0;
  let m: RegExpExecArray | null;
  let i = 0;

  while ((m = patron.exec(texto)) !== null) {
    if (m.index > ultimo) partes.push(texto.slice(ultimo, m.index));
    const t = m[0];
    if (t.startsWith("**")) {
      partes.push(<strong key={`${clave}-b${i}`}>{t.slice(2, -2)}</strong>);
    } else if (t.startsWith("`")) {
      partes.push(<code key={`${clave}-c${i}`}>{t.slice(1, -1)}</code>);
    } else {
      partes.push(
        <span className="cite" key={`${clave}-x${i}`}>
          {t}
        </span>,
      );
    }
    ultimo = m.index + t.length;
    i += 1;
  }
  if (ultimo < texto.length) partes.push(texto.slice(ultimo));
  return partes;
}

export function Answer({ text, streaming }: { text: string; streaming?: boolean }) {
  const bloques: ReactNode[] = [];
  let lista: string[] = [];

  const cerrarLista = (k: number) => {
    if (!lista.length) return;
    bloques.push(
      <ul key={`u${k}`}>
        {lista.map((li, j) => (
          <li key={j}>{inline(li, `l${k}-${j}`)}</li>
        ))}
      </ul>,
    );
    lista = [];
  };

  text.split("\n").forEach((linea, k) => {
    const t = linea.trim();
    if (/^[-*•]\s+/.test(t)) {
      lista.push(t.replace(/^[-*•]\s+/, ""));
      return;
    }
    cerrarLista(k);
    if (!t) return;
    if (/^#{1,4}\s/.test(t)) {
      bloques.push(<h3 key={`h${k}`}>{inline(t.replace(/^#{1,4}\s/, ""), `h${k}`)}</h3>);
    } else {
      bloques.push(<p key={`p${k}`}>{inline(t, `p${k}`)}</p>);
    }
  });
  cerrarLista(999);

  return (
    <div className={`prose-answer text-[15px] ${streaming ? "caret" : ""}`}>
      {bloques.map((b, i) => (
        <Fragment key={i}>{b}</Fragment>
      ))}
    </div>
  );
}
