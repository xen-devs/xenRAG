import { useRef, useState } from "react"
import { useParams } from "react-router-dom"
import toast from "react-hot-toast"

import { UploadZone, type ParsedFile } from "@/components/knowledge-base/upload-zone"
import { ValidationReportPanel } from "@/components/knowledge-base/validation-report"
import { FieldMappingPanel } from "@/components/knowledge-base/field-mapping-panel"
import { IngestionProgress } from "@/components/knowledge-base/ingestion-progress"
import { DatasetVersions } from "@/components/knowledge-base/dataset-versions"
import {
  type UploadResponse,
  type ValidationReport,
  uploadFiles,
  validateUpload,
  applyFieldMapping,
  startIngestion,
} from "@/lib/knowledge-base-api"

type Step = "upload" | "preview" | "validate" | "mapping" | "ingest"

export function KnowledgeBasePage() {
  const { orgId } = useParams()

  // Upload flow state
  const [step, setStep] = useState<Step>("upload")
  const [uploadResult, setUploadResult] = useState<UploadResponse | null>(null)
  const [report, setReport] = useState<ValidationReport | null>(null)
  const [loading, setLoading] = useState(false)

  // Ingestion progress
  const [ingBatch, setIngBatch] = useState(0)
  const [ingTotal, setIngTotal] = useState(0)
  const [ingMsg, setIngMsg] = useState("")
  const [ingComplete, setIngComplete] = useState(false)
  const [ingError, setIngError] = useState(false)
  const [versionRefresh, setVersionRefresh] = useState(0)
  const abortRef = useRef<AbortController | null>(null)


  const handleFilesReady = async (files: ParsedFile[]) => {
    if (!orgId) return
    setLoading(true)
    try {
      const result = await uploadFiles(
        orgId,
        files.map((f) => ({
          filename: f.name,
          content: f.content,
          size: f.size,
        })),
      )
      setUploadResult(result)
      setStep("preview")
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "response" in err
          ? ((err as { response?: { data?: { detail?: string } } }).response
              ?.data?.detail ?? "Upload failed")
          : "Upload failed"
      toast.error(msg)
    } finally {
      setLoading(false)
    }
  }

  const handleValidate = async () => {
    if (!orgId || !uploadResult) return
    setLoading(true)
    try {
      const r = await validateUpload(orgId, uploadResult.upload_id)
      setReport(r)
      setStep("validate")
    } catch {
      toast.error("Validation failed")
    } finally {
      setLoading(false)
    }
  }

  const handleApplyMapping = async (mapping: Record<string, string>) => {
    if (!orgId || !uploadResult) return
    setLoading(true)
    try {
      const r = await applyFieldMapping(orgId, uploadResult.upload_id, mapping)
      setReport(r)
      setStep("validate")
    } catch {
      toast.error("Field mapping failed")
    } finally {
      setLoading(false)
    }
  }

  const handleConfirmIngest = () => {
    if (!orgId || !uploadResult) return
    setStep("ingest")
    setIngBatch(0)
    setIngTotal(0)
    setIngMsg("Starting ingestion...")
    setIngComplete(false)
    setIngError(false)

    const ctrl = startIngestion(
      orgId,
      uploadResult.upload_id,
      (data) => {
        setIngBatch((data.batch as number) || 0)
        setIngTotal((data.totalBatches as number) || 0)
        setIngMsg((data.message as string) || "")
      },
      (data) => {
        setIngComplete(true)
        setVersionRefresh((n) => n + 1)
        setIngMsg((data.message as string) || "Ingestion complete!")
        toast.success("Ingestion complete!")
      },
      (msg) => {
        setIngError(true)
        setIngMsg(msg)
        toast.error(msg.length > 120 ? msg.slice(0, 120) + "..." : msg)
      },
    )
    abortRef.current = ctrl
  }

  const resetUpload = () => {
    if (abortRef.current) abortRef.current.abort()
    setStep("upload")
    setUploadResult(null)
    setReport(null)
    setIngComplete(false)
    setIngError(false)
  }


  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-lg font-semibold">Knowledge Base</h2>
        <p className="text-muted-foreground text-sm">
          Upload customer feedback data to power your RAG retrieval.
        </p>
      </div>

      <section className="space-y-4 rounded-lg border p-5">
        <div>
          <h3 className="text-sm font-semibold">Upload & Ingest</h3>
          <p className="text-muted-foreground text-xs">
            Upload JSON or JSONL files containing customer feedback data.
          </p>
        </div>

        {step === "upload" && (
          <UploadZone onFilesReady={handleFilesReady} isLoading={loading} />
        )}

        {step === "preview" && uploadResult && (
          <div className="space-y-4">
            <h4 className="text-sm font-medium">Upload Preview</h4>
            <div className="text-muted-foreground flex flex-wrap gap-4 text-xs">
              <span>{uploadResult.document_count} records parsed</span>
              <span>{uploadResult.file_metadata.length} file(s)</span>
            </div>
            {uploadResult.file_metadata.map((fm, i) => (
              <div
                key={i}
                className="bg-muted/50 flex flex-wrap items-center gap-3 rounded-md border px-3 py-2 text-xs"
              >
                <span className="font-medium">{fm.filename}</span>
                <span className="text-muted-foreground">
                  {(fm.size / 1024).toFixed(1)} KB
                </span>
                <span className="text-muted-foreground">
                  {fm.row_count} rows
                </span>
              </div>
            ))}

            {uploadResult.preview.length > 0 && (
              <div className="space-y-2">
                <p className="text-muted-foreground text-xs">
                  Sample records (first {uploadResult.preview.length}):
                </p>
                <div className="bg-muted/30 max-h-48 overflow-auto rounded-md border p-3">
                  <pre className="text-xs">
                    {JSON.stringify(uploadResult.preview, null, 2)}
                  </pre>
                </div>
              </div>
            )}

            <div className="flex justify-end gap-2">
              <button
                onClick={resetUpload}
                className="text-muted-foreground hover:text-foreground rounded-md px-3 py-1.5 text-sm transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleValidate}
                disabled={loading}
                className="bg-primary text-primary-foreground hover:bg-primary/90 rounded-md px-4 py-1.5 text-sm font-medium disabled:opacity-50"
              >
                {loading ? "Validating..." : "Validate"}
              </button>
            </div>
          </div>
        )}

        {step === "validate" && report && (
          <ValidationReportPanel
            report={report}
            onMapFields={() => setStep("mapping")}
            onConfirmIngest={handleConfirmIngest}
            onCancel={resetUpload}
          />
        )}

        {step === "mapping" && report && (
          <FieldMappingPanel
            sourceFields={report.source_fields}
            currentMapping={report.detected_mapping}
            onApply={handleApplyMapping}
            onCancel={() => setStep("validate")}
            isLoading={loading}
          />
        )}

        {step === "ingest" && (
          <IngestionProgress
            batch={ingBatch}
            totalBatches={ingTotal}
            message={ingMsg}
            isComplete={ingComplete}
            isError={ingError}
            onReset={resetUpload}
          />
        )}
      </section>

      <section className="space-y-4 rounded-lg border p-5">
        <div>
          <h3 className="text-sm font-semibold">Dataset Versions</h3>
          <p className="text-muted-foreground text-xs">
            Manage and activate dataset versions for this organization.
          </p>
        </div>
        <DatasetVersions refreshTrigger={versionRefresh} />
      </section>
    </div>
  )
}