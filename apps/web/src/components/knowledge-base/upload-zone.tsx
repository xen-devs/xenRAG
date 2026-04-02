import { type DragEvent, useCallback, useRef, useState } from "react"
import { FileUp, X } from "lucide-react"
import { Button } from "@/components/ui/button"

export interface ParsedFile {
  name: string
  size: number
  content: string
}

interface UploadZoneProps {
  onFilesReady: (files: ParsedFile[]) => void
  isLoading?: boolean
}

const ACCEPTED = ".json,.jsonl"
const MAX_FILE_SIZE = 100 * 1024 * 1024 // 100 MB

export function UploadZone({ onFilesReady, isLoading }: UploadZoneProps) {
  const [files, setFiles] = useState<ParsedFile[]>([])
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const readFiles = useCallback(async (fileList: FileList) => {
    setError(null)
    const parsed: ParsedFile[] = []

    for (const file of Array.from(fileList)) {
      const ext = file.name.split(".").pop()?.toLowerCase()
      if (ext !== "json" && ext !== "jsonl") {
        setError(`Unsupported file type: .${ext}. Use .json or .jsonl`)
        continue
      }
      if (file.size > MAX_FILE_SIZE) {
        setError(`File ${file.name} exceeds 100 MB limit`)
        continue
      }

      const content = await file.text()
      parsed.push({ name: file.name, size: file.size, content })
    }

    if (parsed.length > 0) {
      setFiles((prev) => [...prev, ...parsed])
    }
  }, [])

  const handleDrop = useCallback(
    (e: DragEvent) => {
      e.preventDefault()
      setDragOver(false)
      if (e.dataTransfer.files.length > 0) {
        readFiles(e.dataTransfer.files)
      }
    },
    [readFiles],
  )

  const handleFileSelect = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (e.target.files && e.target.files.length > 0) {
        readFiles(e.target.files)
      }
      e.target.value = ""
    },
    [readFiles],
  )

  const removeFile = (index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index))
  }

  const handleProcess = () => {
    if (files.length > 0) {
      onFilesReady(files)
    }
  }

  return (
    <div className="space-y-4">
      <div
        className={`flex flex-col items-center justify-center rounded-lg border-2 border-dashed p-8 transition-colors ${
          dragOver
            ? "border-primary bg-primary/5"
            : "border-muted-foreground/25 hover:border-muted-foreground/50"
        }`}
        onDragOver={(e) => {
          e.preventDefault()
          setDragOver(true)
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") inputRef.current?.click()
        }}
      >
        <FileUp className="text-muted-foreground mb-3 h-10 w-10" />
        <p className="text-sm font-medium">
          Drag & drop files here, or click to browse
        </p>
        <p className="text-muted-foreground mt-1 text-xs">
          Supports .json and .jsonl files (max 100 MB each)
        </p>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPTED}
          multiple
          className="hidden"
          onChange={handleFileSelect}
        />
      </div>

      {error && (
        <p className="text-destructive text-sm">{error}</p>
      )}

      {files.length > 0 && (
        <div className="space-y-2">
          {files.map((file, i) => (
            <div
              key={`${file.name}-${i}`}
              className="bg-muted/50 flex items-center justify-between rounded-md border px-3 py-2"
            >
              <div className="flex items-center gap-3 overflow-hidden">
                <FileUp className="text-muted-foreground h-4 w-4 shrink-0" />
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium">{file.name}</p>
                  <p className="text-muted-foreground text-xs">
                    {(file.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation()
                  removeFile(i)
                }}
                className="text-muted-foreground hover:text-destructive ml-2 shrink-0"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          ))}

          <div className="flex justify-end">
            <Button onClick={handleProcess} disabled={isLoading}>
              {isLoading ? "Processing..." : "Upload & Parse"}
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}