import { apiClient } from "./api"

export interface SignUpPayload {
  email: string
  password: string
  name: string
  org_name?: string
}

export interface SignInPayload {
  email: string
  password: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
}

export interface UserOrg {
  id: string
  name: string
  role: string
  product_name?: string
  description?: string
}

export interface MeResponse {
  id: string
  email: string
  name: string
  orgs: UserOrg[]
}

export interface CreateOrgPayload {
  name: string
  product_name: string
  description: string
}

export async function signUp(data: SignUpPayload): Promise<TokenResponse> {
  const res = await apiClient.post<TokenResponse>("/api/v1/auth/signup", data)
  return res.data
}

export async function signIn(data: SignInPayload): Promise<TokenResponse> {
  const res = await apiClient.post<TokenResponse>("/api/v1/auth/signin", data)
  return res.data
}

export async function getMe(): Promise<MeResponse> {
  const res = await apiClient.get<MeResponse>("/api/v1/auth/me")
  return res.data
}

export async function createOrg(data: CreateOrgPayload): Promise<UserOrg> {
  const res = await apiClient.post<UserOrg>("/api/v1/orgs", data)
  return res.data
}
