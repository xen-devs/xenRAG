import { type FormEvent, useState } from "react"
import { Loader2 } from "lucide-react"
import toast from "react-hot-toast"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { createOrg } from "@/lib/auth-api"

interface CreateOrgFormProps {
  onSuccess: () => void
}

export function CreateOrgForm({ onSuccess }: CreateOrgFormProps) {
  const [name, setName] = useState("")
  const [productName, setProductName] = useState("")
  const [description, setDescription] = useState("")
  const [loading, setLoading] = useState(false)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (name.trim().length < 2) {
      toast.error("Organization name must be at least 2 characters")
      return
    }

    setLoading(true)

    try {
      await createOrg({
        name: name.trim(),
        product_name: productName.trim() || undefined,
        description: description.trim() || undefined,
      })
      toast.success("Organization created successfully!")
      onSuccess()
    } catch (err: unknown) {
      if (
        err &&
        typeof err === "object" &&
        "response" in err &&
        err.response &&
        typeof err.response === "object" &&
        "data" in err.response
      ) {
        const data = err.response.data as { detail?: string }
        toast.error(data.detail ?? "Failed to create organization")
      } else {
        toast.error("Failed to create organization. Please try again.")
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <label htmlFor="org-name" className="text-sm font-medium">
          Organization Name
        </label>
        <Input
          id="org-name"
          type="text"
          placeholder="e.g., Acme Corporation"
          required
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
      </div>

      <div className="flex flex-col gap-2">
        <label htmlFor="product-name" className="text-sm font-medium">
          Product Name
        </label>
        <Input
          id="product-name"
          type="text"
          placeholder="e.g., Fire TV Stick"
          value={productName}
          onChange={(e) => setProductName(e.target.value)}
        />
      </div>

      <div className="flex flex-col gap-2">
        <label htmlFor="description" className="text-sm font-medium">
          Description
        </label>
        <textarea
          id="description"
          className="flex min-h-[80px] w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
          placeholder="Briefly describe your product or what kind of data the AI will be working with..."
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          maxLength={1000}
        />
      </div>

      <div className="pt-2">
        <Button type="submit" className="w-full" disabled={loading}>
          {loading && <Loader2 className="mr-2 size-4 animate-spin" />}
          Create Organization
        </Button>
      </div>
    </form>
  )
}
