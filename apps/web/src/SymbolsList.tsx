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
      <p className="error-block" data-testid="symbols-error">
        Couldn't load the symbols: {error}
      </p>
    );
  }

  if (!grouped) {
    return (
      <p className="text-ink-muted" data-testid="symbols-loading">
        Loading symbols…
      </p>
    );
  }

  return (
    <section data-testid={grouped.length ? "symbols-list" : "symbols-empty"}>
      <h1 className="text-[22px] font-semibold leading-8">Code symbols</h1>
      <p className="mt-1 text-ink-muted">
        {grouped.length === 0
          ? "The repository indexed, but no code symbols were found; it may hold only data or docs."
          : `${(symbols?.length ?? 0).toLocaleString()} functions, classes and config entries across ${grouped.length} file${grouped.length === 1 ? "" : "s"}, as retrieval sees them.`}
      </p>

      {grouped.length > 0 && (
        <div className="card mt-6 divide-y divide-line">
          {grouped.map((file) => (
            <section key={file.path} data-testid="symbols-file" className="px-4 py-3">
              <h2 className="flex items-baseline justify-between gap-3">
                <code data-testid="symbols-file-path" className="truncate font-mono text-[13px] font-medium">
                  {file.path}
                </code>
                <span className="num flex-none text-[12px] text-ink-muted">
                  {file.items.length + file.hidden}
                </span>
              </h2>
              <ul className="mt-1.5">
                {file.items.map((sym) => (
                  <SymbolRow key={`${sym.file_path}:${sym.start_line}`} sym={sym} />
                ))}
              </ul>
              {file.hidden > 0 && (
                <p className="mt-1 text-[12px] text-ink-muted">
                  {file.hidden} more not shown
                </p>
              )}
            </section>
          ))}
        </div>
      )}
    </section>
  );
}

function SymbolRow({ sym }: { sym: CodeSymbol }) {
  return (
    <li
      data-testid="symbol-row"
      className="grid grid-cols-[minmax(0,1fr)_6rem_4rem] items-baseline gap-3 py-0.5"
    >
      <code data-testid="symbol-name" className="truncate font-mono text-[13px]">
        {sym.name}
        {sym.signature && <span className="text-ink-muted">{sym.signature}</span>}
      </code>
      <span className="text-[12px] text-ink-muted">{sym.kind}</span>
      <span className="num text-right text-[12px] text-ink-muted">line {sym.start_line}</span>
    </li>
  );
}
