import { apiClient } from "./api"

export interface FilePayload {
  filename: string
  content: string
  size: number
}

export interface FileMeta {
  filename: string
  size: number
  row_count: number
}

export interface UploadResponse {
  upload_id: string
  document_count: number
  file_metadata: FileMeta[]
  preview: Record<string, unknown>[]
}

export interface ValidationError {
  type: string
  message: string
  fields?: string[]
  count?: string
}

export interface ValidationReport {
  is_valid: boolean
  total_records: number
  errors: ValidationError[]
  warnings: ValidationError[]
  source_fields: string[]
  detected_mapping: Record<string, string>
  target_fields: Record<string, string>
}

export interface Dataset {
  id: string
  version_number: number
  collection_name: string
  document_count: number
  segment_count: number
  is_active: boolean
  status: string
  created_at: string
}

export async function uploadFiles(
  orgId: string,
  files: FilePayload[],
): Promise<UploadResponse> {
  const res = await apiClient.post<UploadResponse>(
    `/api/v1/orgs/${orgId}/knowledge-base/upload`,
    { files },
  )
  return res.data
}

export async function validateUpload(
  orgId: string,
  uploadId: string,
): Promise<ValidationReport> {
  const res = await apiClient.post<ValidationReport>(
    `/api/v1/orgs/${orgId}/knowledge-base/uploads/${uploadId}/validate`,
  )
  return res.data
}

export async function applyFieldMapping(
  orgId: string,
  uploadId: string,
  mapping: Record<string, string>,
): Promise<ValidationReport> {
  const res = await apiClient.post<ValidationReport>(
    `/api/v1/orgs/${orgId}/knowledge-base/uploads/${uploadId}/field-mapping`,
    { mapping },
  )
  return res.data
}

export function startIngestion(
  orgId: string,
  uploadId: string,
  onProgress: (data: Record<string, unknown>) => void,
  onComplete: (data: Record<string, unknown>) => void,
  onError: (message: string) => void,
): AbortController {
  const controller = new AbortController()
  const baseUrl = apiClient.defaults.baseURL?.replace(/\/$/, "") ?? ""
  if (!baseUrl) {
    onError("API base URL is not configured (set VITE_API_URL)")
    return controller
  }

  const token = localStorage.getItem("auth_token")

  fetch(
    `${baseUrl}/api/v1/orgs/${orgId}/knowledge-base/uploads/${uploadId}/ingest`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      signal: controller.signal,
    },
  )
    .then(async (response) => {
      if (!response.ok) {
        const text = await response.text()
        onError(`HTTP ${response.status}: ${text}`)
        return
      }

      const reader = response.body?.getReader()
      if (!reader) {
        onError("No response body")
        return
      }

      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n")
        buffer = lines.pop() || ""

        let eventType = "message"
        for (const line of lines) {
          if (line.startsWith("event: ")) {
            eventType = line.slice(7).trim()
          } else if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6))
              if (eventType === "complete") {
                onComplete(data)
              } else if (eventType === "error") {
                onError(data.message || "Unknown error")
              } else {
                onProgress(data)
              }
            } catch {
              // skip malformed lines
            }
            eventType = "message"
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== "AbortError") {
        onError(err.message || "Connection failed")
      }
    })

  return controller
}

export async function listDatasets(orgId: string): Promise<Dataset[]> {
  const res = await apiClient.get<{ datasets: Dataset[] }>(
    `/api/v1/orgs/${orgId}/knowledge-base/datasets`,
  )
  return res.data.datasets
}

export async function activateDataset(
  orgId: string,
  datasetId: string,
): Promise<Dataset> {
  const res = await apiClient.post<Dataset>(
    `/api/v1/orgs/${orgId}/knowledge-base/datasets/${datasetId}/activate`,
  )
  return res.data
}

export async function deleteDataset(
  orgId: string,
  datasetId: string,
): Promise<void> {
  await apiClient.delete(
    `/api/v1/orgs/${orgId}/knowledge-base/datasets/${datasetId}`,
  )
}