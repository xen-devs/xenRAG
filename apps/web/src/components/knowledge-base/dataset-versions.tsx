import { useCallback, useEffect, useState } from "react"
import { useParams } from "react-router-dom"
import { CheckCircle2, Loader2, Trash2 } from "lucide-react"
import toast from "react-hot-toast"
import { Button } from "@/components/ui/button"
import { ConfirmModal } from "@/components/ui/confirm-modal"
import {
  type Dataset,
  activateDataset,
  deleteDataset,
  listDatasets,
} from "@/lib/knowledge-base-api"

interface DatasetVersionsProps {
  refreshTrigger: number
}

export function DatasetVersions({ refreshTrigger }: DatasetVersionsProps) {
  const { orgId } = useParams()
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [actionLoading, setActionLoading] = useState<string | null>(null)
  const [deleteTarget, setDeleteTarget] = useState<Dataset | null>(null)
  const [deleteLoading, setDeleteLoading] = useState(false)

  const load = useCallback(async () => {
    if (!orgId) return
    setLoading(true)
    try {
      const data = await listDatasets(orgId)
      setDatasets(data)
    } catch {
      toast.error("Failed to load datasets")
    } finally {
      setLoading(false)
    }
  }, [orgId])

  useEffect(() => {
    load()
  }, [load, refreshTrigger])

  const handleActivate = async (datasetId: string) => {
    if (!orgId) return
    setActionLoading(datasetId)
    try {
      await activateDataset(orgId, datasetId)
      toast.success("Dataset activated")
      await load()
    } catch {
      toast.error("Failed to activate dataset")
    } finally {
      setActionLoading(null)
    }
  }

  const handleDelete = async () => {
    if (!orgId || !deleteTarget) return

    setDeleteLoading(true)
    try {
      await deleteDataset(orgId, deleteTarget.id)
      toast.success("Dataset deleted")
      await load()
      setDeleteTarget(null)
    } catch {
      toast.error("Failed to delete dataset")
    } finally {
      setDeleteLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Loader2 className="text-muted-foreground h-5 w-5 animate-spin" />
      </div>
    )
  }

  if (datasets.length === 0) {
    return (
      <div className="text-muted-foreground rounded-lg border border-dashed py-8 text-center text-sm">
        No datasets yet. Upload and ingest data to create your first version.
      </div>
    )
  }

  return (
    <div className="rounded-md border">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b">
            <th className="text-muted-foreground px-3 py-2 text-left text-xs font-medium">
              Version
            </th>
            <th className="text-muted-foreground px-3 py-2 text-left text-xs font-medium">
              Documents
            </th>
            <th className="text-muted-foreground px-3 py-2 text-left text-xs font-medium">
              Segments
            </th>
            <th className="text-muted-foreground px-3 py-2 text-left text-xs font-medium">
              Status
            </th>
            <th className="text-muted-foreground px-3 py-2 text-left text-xs font-medium">
              Created
            </th>
            <th className="text-muted-foreground px-3 py-2 text-right text-xs font-medium">
              Actions
            </th>
          </tr>
        </thead>
        <tbody>
          {datasets.map((ds) => (
            <tr key={ds.id} className="border-b last:border-b-0">
              <td className="px-3 py-2 font-mono text-xs">
                v{ds.version_number}
              </td>
              <td className="text-muted-foreground px-3 py-2 text-xs">
                {ds.document_count}
              </td>
              <td className="text-muted-foreground px-3 py-2 text-xs">
                {ds.segment_count}
              </td>
              <td className="px-3 py-2">
                <span
                  className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${
                    ds.is_active
                      ? "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400"
                      : ds.status === "failed"
                        ? "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400"
                        : "bg-muted text-muted-foreground"
                  }`}
                >
                  {ds.is_active && (
                    <CheckCircle2 className="h-3 w-3" />
                  )}
                  {ds.is_active ? "Active" : ds.status}
                </span>
              </td>
              <td className="text-muted-foreground px-3 py-2 text-xs">
                {new Date(ds.created_at).toLocaleDateString()}
              </td>
              <td className="px-3 py-2 text-right">
                <div className="flex items-center justify-end gap-1">
                  {!ds.is_active && ds.status !== "failed" && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="h-7 text-xs"
                      disabled={actionLoading === ds.id}
                      onClick={() => handleActivate(ds.id)}
                    >
                      {actionLoading === ds.id ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        "Activate"
                      )}
                    </Button>
                  )}
                  {!ds.is_active && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="text-destructive hover:bg-destructive/10 h-7"
                      disabled={actionLoading === ds.id}
                      onClick={() => setDeleteTarget(ds)}
                    >
                      <Trash2 className="h-3 w-3" />
                    </Button>
                  )}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <ConfirmModal
        open={!!deleteTarget}
        onOpenChange={(open) => { if (!open) setDeleteTarget(null) }}
        title="Delete dataset?"
        description="This dataset version will be permanently deleted."
        confirmLabel="Delete"
        variant="destructive"
        isLoading={deleteLoading}
        onConfirm={handleDelete}
      />
    </div>
  )
}