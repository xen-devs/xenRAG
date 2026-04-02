import { useState } from "react"
import { Button } from "@/components/ui/button"

const TARGET_OPTIONS = [
  { value: "", label: "-- skip --" },
  { value: "text", label: "text (main content)" },
  { value: "id", label: "id (unique identifier)" },
  { value: "rating", label: "rating (numeric score)" },
  { value: "user_id", label: "user_id (customer ID)" },
  { value: "title", label: "title (document title)" },
  { value: "category", label: "category (topic/type)" },
]

interface FieldMappingPanelProps {
  sourceFields: string[]
  currentMapping: Record<string, string>
  onApply: (mapping: Record<string, string>) => void
  onCancel: () => void
  isLoading?: boolean
}

export function FieldMappingPanel({
  sourceFields,
  currentMapping,
  onApply,
  onCancel,
  isLoading,
}: FieldMappingPanelProps) {
  const [mapping, setMapping] = useState<Record<string, string>>(() => {
    const initial: Record<string, string> = {}
    for (const field of sourceFields) {
      initial[field] = currentMapping[field] || ""
    }
    return initial
  })

  const handleChange = (sourceField: string, targetField: string) => {
    setMapping((prev) => {
      const next = { ...prev }
      if (targetField === "") {
        delete next[sourceField]
      } else {
        for (const [src, tgt] of Object.entries(next)) {
          if (tgt === targetField && src !== sourceField) {
            delete next[src]
          }
        }
        next[sourceField] = targetField
      }
      return next
    })
  }

  const handleApply = () => {
    const cleaned: Record<string, string> = {}
    for (const [src, tgt] of Object.entries(mapping)) {
      if (tgt) cleaned[src] = tgt
    }
    onApply(cleaned)
  }

  const hasTextMapping = Object.values(mapping).includes("text")

  return (
    <div className="space-y-4">
      <p className="text-muted-foreground text-sm">
        Map your file fields to the target schema. At minimum, one field must
        map to <span className="font-mono font-medium">text</span>.
      </p>

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
            {sourceFields.map((field) => (
              <tr key={field} className="border-b last:border-b-0">
                <td className="px-3 py-2 font-mono text-xs">{field}</td>
                <td className="px-3 py-2">
                  <select
                    value={mapping[field] || ""}
                    onChange={(e) => handleChange(field, e.target.value)}
                    className="bg-background w-full rounded-md border px-2 py-1 text-xs"
                  >
                    {TARGET_OPTIONS.map((opt) => (
                      <option key={opt.value} value={opt.value}>
                        {opt.label}
                      </option>
                    ))}
                  </select>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {!hasTextMapping && (
        <p className="text-destructive text-xs">
          You must map at least one field to "text" (main content).
        </p>
      )}

      <div className="flex justify-end gap-2">
        <Button variant="outline" onClick={onCancel}>
          Back
        </Button>
        <Button
          onClick={handleApply}
          disabled={!hasTextMapping || isLoading}
        >
          {isLoading ? "Applying..." : "Apply Mapping"}
        </Button>
      </div>
    </div>
  )
}