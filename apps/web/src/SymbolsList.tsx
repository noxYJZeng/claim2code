import { useEffect, useMemo, useState } from "react";
import { getSymbols, type CodeSymbol } from "./api";

type SymbolsListProps = {
  repositoryId: string;
};

// SymbolsList — Phase 3 extracted-symbols viewer.
//
// Fetches GET /repositories/{repository_id}/symbols and groups the
// flat list by file_path. v1 caps each file at MAX_PER_FILE rows
// (with a "+N more" footer) so a 10k-symbol repo doesn't paint the
// DOM into oblivion; Phase 5+ adds cursor pagination at the API.
//
// Empty/error/loading states mirror ClaimsList so the two sections
// feel symmetric.
const MAX_PER_FILE = 50;

export default function SymbolsList({ repositoryId }: SymbolsListProps) {
  const [symbols, setSymbols] = useState<CodeSymbol[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    getSymbols(repositoryId, controller.signal).then((result) => {
      if (result.ok) {
        setSymbols(result.response.symbols);
        setError(null);
      } else {
        setSymbols(null);
        setError(result.error);
      }
    });
    return () => controller.abort();
  }, [repositoryId]);

  // Group symbols by file_path, preserving the API's ordering (already
  // sorted by file_path, start_line on the server side).
  const grouped = useMemo(() => {
    if (!symbols) return null;
    const out: { path: string; items: CodeSymbol[]; hidden: number }[] = [];
    let bucket: { path: string; items: CodeSymbol[]; hidden: number } | null =
      null;
    for (const sym of symbols) {
      if (!bucket || bucket.path !== sym.file_path) {
        bucket = { path: sym.file_path, items: [], hidden: 0 };
        out.push(bucket);
      }
      if (bucket.items.length < MAX_PER_FILE) {
        bucket.items.push(sym);
      } else {
        bucket.hidden += 1;
      }
    }
    return out;
  }, [symbols]);

  if (error) {
    return (
      <section
        className="rounded-2xl border border-rose-200 bg-rose-50 p-6 text-sm text-rose-800"
        data-testid="symbols-error"
      >
        Failed to load symbols: {error}
      </section>
    );
  }

  if (!grouped) {
    return (
      <section
        className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-500"
        data-testid="symbols-loading"
      >
        Loading symbols…
      </section>
    );
  }

  if (grouped.length === 0) {
    return (
      <section
        className="rounded-2xl border border-slate-200 bg-white p-6 text-sm text-slate-600"
        data-testid="symbols-empty"
      >
        The repo indexed but no code symbols were extracted (it might
        only contain data / docs files).
      </section>
    );
  }

  return (
    <section
      className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm"
      data-testid="symbols-list"
    >
      <div className="mb-4 flex items-baseline justify-between">
        <h2 className="text-lg font-medium">Extracted symbols</h2>
        <span className="text-sm text-slate-500">
          {symbols?.length ?? 0} symbol{(symbols?.length ?? 0) === 1 ? "" : "s"}{" "}
          across {grouped.length} file{grouped.length === 1 ? "" : "s"}
        </span>
      </div>

      <ul className="space-y-4">
        {grouped.map((file) => (
          <li
            key={file.path}
            data-testid="symbols-file"
            className="rounded-xl border border-slate-200 bg-slate-50 p-4"
          >
            <div className="mb-2 flex items-baseline justify-between gap-2">
              <code
                data-testid="symbols-file-path"
                className="text-xs font-medium text-slate-700"
              >
                {file.path}
              </code>
              <span className="text-xs text-slate-500">
                {file.items.length + file.hidden} symbol
                {file.items.length + file.hidden === 1 ? "" : "s"}
              </span>
            </div>
            <ul className="space-y-1">
              {file.items.map((sym) => (
                <SymbolRow key={`${sym.file_path}:${sym.start_line}`} sym={sym} />
              ))}
            </ul>
            {file.hidden > 0 && (
              <p className="mt-2 text-xs text-slate-500">
                + {file.hidden} more symbol{file.hidden === 1 ? "" : "s"} not
                shown
              </p>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}

function SymbolRow({ sym }: { sym: CodeSymbol }) {
  return (
    <li
      data-testid="symbol-row"
      className="flex items-baseline gap-3 text-sm"
    >
      <span className="shrink-0 rounded-full bg-slate-900 px-2 py-0.5 text-xs font-medium uppercase tracking-wide text-white">
        {sym.kind}
      </span>
      <span className="shrink-0 rounded-full bg-slate-200 px-2 py-0.5 text-xs font-medium uppercase tracking-wide text-slate-700">
        {sym.language}
      </span>
      <code
        data-testid="symbol-name"
        className="font-mono text-sm text-slate-900"
      >
        {sym.name}
        {sym.signature && (
          <span className="text-slate-500">{sym.signature}</span>
        )}
      </code>
      <span className="shrink-0 text-xs text-slate-400">
        L{sym.start_line}:{sym.start_col}
      </span>
    </li>
  );
}