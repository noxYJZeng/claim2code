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
      <p className="error-note" data-testid="symbols-error">
        Failed to load symbols: {error}
      </p>
    );
  }

  if (!grouped) {
    return (
      <p className="text-sm text-ink-3" data-testid="symbols-loading">
        Loading symbols…
      </p>
    );
  }

  if (grouped.length === 0) {
    return (
      <p className="text-sm text-ink-2" data-testid="symbols-empty">
        The repo indexed but no code symbols were extracted (it might only contain
        data or docs files).
      </p>
    );
  }

  return (
    <section data-testid="symbols-list">
      <p className="text-sm text-ink-2">
        {symbols?.length ?? 0} symbol{(symbols?.length ?? 0) === 1 ? "" : "s"} across{" "}
        {grouped.length} file{grouped.length === 1 ? "" : "s"}.
      </p>

      <ul className="mt-6 space-y-6">
        {grouped.map((file) => (
          <li key={file.path} data-testid="symbols-file">
            <div className="flex items-baseline justify-between gap-2 border-b border-rule pb-1.5">
              <code data-testid="symbols-file-path" className="font-mono text-sm font-medium">
                {file.path}
              </code>
              <span className="text-xs text-ink-3">
                {file.items.length + file.hidden} symbol
                {file.items.length + file.hidden === 1 ? "" : "s"}
              </span>
            </div>
            <ul className="mt-2 space-y-1">
              {file.items.map((sym) => (
                <SymbolRow key={`${sym.file_path}:${sym.start_line}`} sym={sym} />
              ))}
            </ul>
            {file.hidden > 0 && (
              <p className="mt-2 text-xs text-ink-3">
                + {file.hidden} more symbol{file.hidden === 1 ? "" : "s"} not shown
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
      className="grid grid-cols-[3.5rem_5rem_minmax(0,1fr)] items-baseline gap-3 text-sm"
    >
      <span className="text-right font-mono text-xs text-ink-3">L{sym.start_line}</span>
      <span className="text-xs text-ink-3">
        {sym.kind} · {sym.language}
      </span>
      <code data-testid="symbol-name" className="truncate font-mono text-[13px]">
        {sym.name}
        {sym.signature && <span className="text-ink-3">{sym.signature}</span>}
      </code>
    </li>
  );
}
