import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "LipTone Studio",
  description: "Responsive virtual lipstick try-on powered by Next.js and FastAPI.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
