import { useMemo, useState, type ReactNode } from 'react';
import {
  flexRender, getCoreRowModel, getFilteredRowModel, getSortedRowModel, useReactTable,
  type ColumnDef, type SortingState, type VisibilityState,
} from '@tanstack/react-table';
import * as Menu from '@radix-ui/react-dropdown-menu';
import { ArrowDown, ArrowUp, Check, Columns3, Search } from 'lucide-react';
import { cn } from './ui';

/* eslint-disable @typescript-eslint/no-explicit-any */
export type Col<T> = ColumnDef<T, any> & { meta?: { num?: boolean; mono?: boolean; hide?: boolean; wrap?: boolean } };

/** One table component for every table: sortable headers, global filter, column visibility, sticky header, row click. */
export function DataTable<T>({ data, columns, onRowClick, initialSort, maxHeight = 440, search = true, toolbar, empty = 'No rows', dense = true }: {
  data: T[]; columns: Col<T>[]; onRowClick?: (row: T) => void; initialSort?: SortingState; maxHeight?: number | 'none'; search?: boolean;
  toolbar?: ReactNode; empty?: ReactNode; dense?: boolean;
}) {
  const [sorting, setSorting] = useState<SortingState>(initialSort ?? []);
  const [filter, setFilter] = useState('');
  const [vis, setVis] = useState<VisibilityState>(() => Object.fromEntries(columns.filter(c => c.meta?.hide).map(c => [String((c as any).id ?? (c as any).accessorKey), false])));
  const table = useReactTable({
    data, columns, state: { sorting, globalFilter: filter, columnVisibility: vis },
    onSortingChange: setSorting, onGlobalFilterChange: setFilter, onColumnVisibilityChange: setVis,
    getCoreRowModel: getCoreRowModel(), getSortedRowModel: getSortedRowModel(), getFilteredRowModel: getFilteredRowModel(),
    globalFilterFn: 'includesString',
  });
  const rows = table.getRowModel().rows;
  const hideable = useMemo(() => table.getAllLeafColumns().filter(c => c.getCanHide()), [table]);
  return (
    <div className="min-w-0">
      {(search || toolbar || hideable.length > 4) && (
        <div className="mb-2 flex flex-wrap items-center gap-2">
          {search && (
            <label className="relative">
              <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted" />
              <input value={filter} onChange={e => setFilter(e.target.value)} placeholder="Filter…" aria-label="Filter rows"
                className="h-8 w-48 rounded-lg border border-line bg-panel pl-8 pr-2 text-[12.5px] outline-none placeholder:text-muted focus:ring-2 focus:ring-accent/30" />
            </label>
          )}
          {toolbar}
          <span className="ml-auto text-[11.5px] text-muted tnum">{rows.length} of {data.length}</span>
          {hideable.length > 4 && (
            <Menu.Root>
              <Menu.Trigger asChild>
                <button type="button" className="inline-flex h-8 items-center gap-1.5 rounded-lg border border-line bg-panel px-2.5 text-[12px] text-ink-2 hover:text-ink">
                  <Columns3 className="size-3.5" /> Columns
                </button>
              </Menu.Trigger>
              <Menu.Portal>
                <Menu.Content align="end" sideOffset={6} className="z-50 max-h-80 min-w-44 overflow-auto rounded-xl border border-line bg-panel p-1 shadow-xl">
                  {hideable.map(c => (
                    <Menu.CheckboxItem key={c.id} checked={c.getIsVisible()} onCheckedChange={v => c.toggleVisibility(!!v)} onSelect={e => e.preventDefault()}
                      className="flex cursor-pointer select-none items-center gap-2 rounded-md px-2 py-1.5 text-[12.5px] text-ink-2 outline-none data-[highlighted]:bg-panel-2 data-[highlighted]:text-ink">
                      <span className="flex size-3.5 items-center justify-center">{c.getIsVisible() && <Check className="size-3.5" />}</span>
                      {typeof c.columnDef.header === 'string' ? c.columnDef.header : c.id}
                    </Menu.CheckboxItem>
                  ))}
                </Menu.Content>
              </Menu.Portal>
            </Menu.Root>
          )}
        </div>
      )}
      <div className="overflow-auto rounded-xl border border-line" style={{ maxHeight: maxHeight === 'none' ? undefined : maxHeight }}>
        <table className="w-full border-separate border-spacing-0 text-[12.5px] tnum">
          <thead>
            {table.getHeaderGroups().map(hg => (
              <tr key={hg.id}>
                {hg.headers.map(h => {
                  const m = (h.column.columnDef as Col<T>).meta;
                  const sorted = h.column.getIsSorted();
                  return (
                    <th key={h.id} scope="col"
                      className={cn('sticky top-0 z-[1] whitespace-nowrap border-b border-line bg-panel-2 px-3 py-2 text-left text-[11.5px] font-semibold text-ink-2',
                        m?.num && 'text-right', h.column.getCanSort() && 'cursor-pointer select-none hover:text-ink')}
                      onClick={h.column.getToggleSortingHandler()} aria-sort={sorted === 'asc' ? 'ascending' : sorted === 'desc' ? 'descending' : 'none'}>
                      <span className={cn('inline-flex items-center gap-1', m?.num && 'flex-row-reverse')}>
                        {flexRender(h.column.columnDef.header, h.getContext())}
                        {sorted === 'asc' ? <ArrowUp className="size-3" /> : sorted === 'desc' ? <ArrowDown className="size-3" /> : null}
                      </span>
                    </th>
                  );
                })}
              </tr>
            ))}
          </thead>
          <tbody>
            {rows.length === 0 && (
              <tr><td colSpan={table.getVisibleLeafColumns().length} className="px-3 py-8 text-center text-muted">{empty}</td></tr>
            )}
            {rows.map(r => (
              <tr key={r.id} onClick={onRowClick ? () => onRowClick(r.original) : undefined}
                className={cn('group', onRowClick && 'cursor-pointer')}>
                {r.getVisibleCells().map(c => {
                  const m = (c.column.columnDef as Col<T>).meta;
                  return (
                    <td key={c.id} className={cn('border-b border-line px-3 text-ink group-last:border-b-0 group-hover:bg-panel-2/70', dense ? 'py-1.5' : 'py-2.5',
                      m?.num && 'text-right', m?.mono && 'font-mono text-[11.5px] text-ink-2', m?.wrap ? 'min-w-56 whitespace-normal' : 'whitespace-nowrap')}>
                      {flexRender(c.column.columnDef.cell, c.getContext())}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** helper: a numeric column */
export function numCol<T>(id: string, header: string, get: (r: T) => number | null | undefined, fmt: (v: number | null | undefined) => ReactNode, extra: Partial<Col<T>> = {}): Col<T> {
  return {
    id, header, accessorFn: r => get(r) ?? null, sortUndefined: 'last', meta: { num: true, ...(extra.meta ?? {}) },
    cell: c => fmt(c.getValue() as number | null),
    sortingFn: (a, b, cid) => ((a.getValue(cid) as number) ?? -Infinity) - ((b.getValue(cid) as number) ?? -Infinity),
    ...extra,
  } as Col<T>;
}
