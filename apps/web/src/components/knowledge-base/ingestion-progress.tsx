import { CheckCircle2, Loader2, XCircle } from "lucide-react"
import { Button } from "@/components/ui/button"

interface IngestionProgressProps {
  batch: number
  totalBatches: number
  message: string
  isComplete: boolean
  isError: boolean
  onReset: () => void
}

export function IngestionProgress({
  batch,
  totalBatches,
  message,
  isComplete,
  isError,
  onReset,
}: IngestionProgressProps) {
  const pct =
    totalBatches > 0 ? Math.round((batch / totalBatches) * 100) : 0

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="text-muted-foreground">
            {isComplete
              ? "Complete"
              : isError
                ? "Failed"
                : `Batch ${batch} / ${totalBatches}`}
          </span>
          <span className="text-muted-foreground font-mono">{pct}%</span>
        </div>
        <div className="bg-muted h-2 w-full overflow-hidden rounded-full">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              isError
                ? "bg-destructive"
                : isComplete
                  ? "bg-green-500"
                  : "bg-primary"
            }`}
            style={{ width: `${isComplete ? 100 : pct}%` }}
          />
        </div>
      </div>

      <div className="flex items-start gap-2">
        {isComplete ? (
          <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-green-500" />
        ) : isError ? (
          <XCircle className="text-destructive mt-0.5 h-4 w-4 shrink-0" />
        ) : (
          <Loader2 className="text-primary mt-0.5 h-4 w-4 shrink-0 animate-spin" />
        )}
        <p
          className={`text-sm ${
            isError
              ? "text-destructive"
              : isComplete
                ? "text-green-600 dark:text-green-400"
                : "text-muted-foreground"
          }`}
        >
          {message}
        </p>
      </div>

      {(isComplete || isError) && (
        <div className="flex justify-end">
          <Button onClick={onReset}>Upload Another</Button>
        </div>
      )}
    </div>
  )
}