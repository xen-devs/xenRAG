import { AlertCircle, AlertTriangle, CheckCircle2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { ValidationReport as ValidationReportType } from "@/lib/knowledge-base-api"

interface ValidationReportProps {
  report: ValidationReportType
  onMapFields: () => void
  onConfirmIngest: () => void
  onCancel: () => void
}

export function ValidationReportPanel({
  report,
  onMapFields,
  onConfirmIngest,
  onCancel,
}: ValidationReportProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        {report.is_valid ? (
          <CheckCircle2 className="h-5 w-5 shrink-0 text-green-500" />
        ) : (
          <AlertCircle className="h-5 w-5 shrink-0 text-red-500" />
        )}
        <div>
          <p className="text-sm font-medium">
            {report.is_valid
              ? "Validation passed"
              : "Validation failed - fix errors before ingesting"}
          </p>
          <p className="text-muted-foreground text-xs">
            {report.total_records} record(s) found
          </p>
        </div>
      </div>

      {report.errors.length > 0 && (
        <div className="space-y-2">
          {report.errors.map((err, i) => (
            <div
              key={i}
              className="flex items-start gap-2 rounded-md border border-red-200 bg-red-50 p-3 dark:border-red-900 dark:bg-red-950/30"
            >
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-500" />
              <p className="text-sm text-red-700 dark:text-red-400">
                {err.message}
              </p>
            </div>
          ))}
        </div>
      )}

      {report.warnings.length > 0 && (
        <div className="space-y-2">
          {report.warnings.map((warn, i) => (
            <div
              key={i}
              className="flex items-start gap-2 rounded-md border border-yellow-200 bg-yellow-50 p-3 dark:border-yellow-900 dark:bg-yellow-950/30"
            >
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-yellow-600" />
              <p className="text-sm text-yellow-700 dark:text-yellow-400">
                {warn.message}
              </p>
            </div>
          ))}
        </div>
      )}

      {Object.keys(report.detected_mapping).length > 0 && (
        <div className="space-y-2">
          <p className="text-sm font-medium">Field Mapping (auto-detected)</p>
          <div className="rounded-md border">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b">
                  <th className="text-muted-foreground px-3 py-2 text-left text-xs font-medium">
                    Source Field
                  </th>
                  <th className="text-muted-foreground px-3 py-2 text-left text-xs font-medium">
                    Maps To
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(report.detected_mapping).map(([src, tgt]) => (
                  <tr key={src} className="border-b last:border-b-0">
                    <td className="px-3 py-2 font-mono text-xs">{src}</td>
                    <td className="px-3 py-2 font-mono text-xs">{tgt}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {report.source_fields.length > 0 && (
        <div>
          <p className="text-muted-foreground mb-1 text-xs">
            Available source fields:{" "}
            <span className="font-mono">
              {report.source_fields.join(", ")}
            </span>
          </p>
        </div>
      )}

      <div className="flex justify-end gap-2 pt-2">
        <Button variant="outline" onClick={onCancel}>
          Cancel
        </Button>
        <Button variant="outline" onClick={onMapFields}>
          Edit Field Mapping
        </Button>
        <Button onClick={onConfirmIngest} disabled={!report.is_valid}>
          Start Ingestion
        </Button>
      </div>
    </div>
  )
}