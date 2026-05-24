export type ShadeOption = {
  name: string;
  hex: string;
  undertone: string;
  depth: string;
};

export type ShadeCatalog = Record<string, Record<string, ShadeOption[]>>;

export type Recommendation = {
  family: string;
  subcategory: string;
  name: string;
  hex: string;
  undertone: string;
  depth: string;
};

export type ToneProfile = {
  undertone: string;
  depth: string;
  skin_hex: string | null;
};

export type TryOnResponse = {
  detected: boolean;
  image_base64?: string | null;
  tuned_hex?: string | null;
  tone_profile?: ToneProfile | null;
  recommendations: Recommendation[];
  message?: string | null;
};

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

export async function fetchShades(): Promise<ShadeCatalog> {
  const response = await fetch(`${API_BASE}/api/shades`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error("Failed to load shades");
  }
  const data = await response.json();
  return data.shades as ShadeCatalog;
}

export async function submitTryOn(input: {
  file: File;
  shadeName?: string;
  customHex?: string;
  opacity: number;
  finish: string;
}): Promise<TryOnResponse> {
  const formData = new FormData();
  formData.append("file", input.file);
  if (input.shadeName) {
    formData.append("shade_name", input.shadeName);
  }
  if (input.customHex) {
    formData.append("custom_hex", input.customHex);
  }
  formData.append("opacity", input.opacity.toString());
  formData.append("finish", input.finish);

  const response = await fetch(`${API_BASE}/api/try-on`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Try-on request failed");
  }

  return (await response.json()) as TryOnResponse;
}
